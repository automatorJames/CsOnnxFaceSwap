
using FaceSwapOnnx.HelperMathClasses;
using FaceSwapOnnx.OnnxModels;

/// <summary>
/// Requires the following models to have operated on Face:
/// RetinaFace (for BoundingBox and KeyFacePoints)
/// ArcFace (for embedding)
/// </summary>
/// 

public class InSwapper(FaceSwapperOptions options, JobStatsCollector jobStatsCollector) : OnnxModelBase(GetModelConfig(options), jobStatsCollector)
{
    static Mat emap = Static.GetBinaryMatrix(BinaryMatrix.EMap);

    static ModelConfig GetModelConfig(FaceSwapperOptions options) => new ModelConfig(
            OnnxModel.InSwapper,
            options,
            inputMean: 0.0,
            inputStandard: 255.0
        );

    public Tensor<float> Forward(Tensor<float> img, Tensor<float> latent)
    {
        // Normalize img tensor
        float[] imgArray = img.ToArray();
        Span<float> imgSpan = new Span<float>(imgArray);
        for (int i = 0; i < imgSpan.Length; i++)
        {
            imgSpan[i] = (imgSpan[i] - (float)this._modelConfig.InputMean) / (float)this._modelConfig.InputStandard;
        }

        // Prepare inputs
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(this._modelConfig.InputNames[0], img),
            NamedOnnxValue.CreateFromTensor(this._modelConfig.InputNames[1], latent)
        };

        using var results = this._session.Run(inputs);
        var predTensor = results.First().AsTensor<float>();

        return predTensor;
    }

    public Mat Get(Mat inputImage, Face targetFace, Face sourceFace)
    {
        if (_options.TrackStatistics)
        {
            _jobStatsCollector.SignalJobStart(sourceFace.JobId, sourceFace.FaceId, GetType());
            _jobStatsCollector.SignalJobStart(targetFace.JobId, targetFace.FaceId, GetType());
        }

        var keyFacePoints = ExtractKeyFacePointsFromFace(targetFace);
        var (alignedImage, normalizationMatrix) = AlignFace(inputImage, keyFacePoints);
        var imageBlob = CreateImageBlob(alignedImage);

        var latentEmbedding = PrepareLatentEmbedding(sourceFace);
        var predictedTensor = RunForwardModel(imageBlob, latentEmbedding);

        var fakeImage = ConvertPredictionsToImage(predictedTensor);
        var finalMergedImage = WarpAndMergeFace(inputImage, alignedImage, normalizationMatrix, fakeImage);

        if (_options.TrackStatistics)
        {
            _jobStatsCollector.SignalJobEnd(sourceFace.JobId, sourceFace.FaceId, GetType());
            _jobStatsCollector.SignalJobEnd(targetFace.JobId, targetFace.FaceId, GetType());
        }

        return finalMergedImage;
    }

    Mat ExtractKeyFacePointsFromFace(Face face)
    {
        var keyFaceFeaturePointsArray = face.KeyFaceFeatures.PointsFlattened;
        var keyPointCount = keyFaceFeaturePointsArray.Length / 2;
        var keyFacePointsMat = new Mat(keyPointCount, 2, MatType.CV_64F);

        for (int i = 0; i < keyPointCount; i++)
        {
            keyFacePointsMat.Set(i, 0, keyFaceFeaturePointsArray[i * 2]);     // X-coordinate
            keyFacePointsMat.Set(i, 1, keyFaceFeaturePointsArray[i * 2 + 1]); // Y-coordinate
        }

        keyFacePointsMat.ConvertTo(keyFacePointsMat, MatType.CV_32FC1);
        return keyFacePointsMat;
    }

    (Mat alignedImage, Mat normalizationMatrix) AlignFace(Mat inputImage, Mat keyFacePoints)
    {
        // Perform face alignment
        var (alignedImage, normedMat) = FaceAlign.NormCrop2(inputImage, keyFacePoints, _modelConfig.InputSize.Item1);
        return (alignedImage, normedMat);
    }

    Mat CreateImageBlob(Mat alignedImage)
    {
        var blob = CvDnn.BlobFromImage(
            image: alignedImage,
            scaleFactor: 1.0f / (float)_modelConfig.InputStandard,
            size: new Size(_modelConfig.InputSize.Item1, _modelConfig.InputSize.Item2),
            mean: new Scalar(_modelConfig.InputMean, _modelConfig.InputMean, _modelConfig.InputMean),
            swapRB: true);

        return blob;
    }

    DenseTensor<float> PrepareLatentEmbedding(Face sourceFace)
    {
        var latentEmbedding = sourceFace.GetNormedEmbedding();
        latentEmbedding.ConvertTo(latentEmbedding, MatType.CV_32FC1);

        // Multiply latent with emap using Cv2.Gemm
        Cv2.Gemm(
            src1: latentEmbedding.T(),
            src2: emap,
            alpha: 1.0,
            src3: Mat.Zeros(0, 0, latentEmbedding.Type()),
            gamma: 0.0,
            dst: latentEmbedding);

        latentEmbedding = latentEmbedding / latentEmbedding.L2Norm();

        var latentData = latentEmbedding.MatToFlat().Select(d => (float)d).ToArray();
        var latentTensor = new DenseTensor<float>(latentData, new[] { 1, latentData.Length });
        return latentTensor;
    }

    Tensor<float> RunForwardModel(Mat imageBlob, DenseTensor<float> latentTensor)
    {
        var imageTensor = imageBlob.ToDenseTensor();
        var predTensor = Forward(imageTensor, latentTensor);
        return predTensor;
    }

    Mat ConvertPredictionsToImage(Tensor<float> predTensor)
    {
        var predData = predTensor.ToArray();
        var predShape = predTensor.Dimensions.ToArray(); // [1, 3, H, W]

        var batchSize = predShape[0];
        var channelCount = predShape[1];
        var height = predShape[2];
        var width = predShape[3];

        if (batchSize != 1)
        {
            throw new Exception("Expected batch size of 1.");
        }

        var channelMats = new Mat[channelCount];
        var imageSize = height * width;
        for (int c = 0; c < channelCount; c++)
        {
            var channelData = new float[imageSize];
            Array.Copy(predData, c * imageSize, channelData, 0, imageSize);

            var channelMat = new Mat(height, width, MatType.CV_32F);
            Marshal.Copy(channelData, 0, channelMat.Data, channelData.Length);

            channelMats[c] = channelMat;
        }

        var fakeImage = new Mat();
        Cv2.Merge(channelMats, fakeImage);

        fakeImage = fakeImage * 255.0;
        fakeImage.ConvertTo(fakeImage, MatType.CV_8UC3);
        Cv2.CvtColor(fakeImage, fakeImage, ColorConversionCodes.RGB2BGR);

        return fakeImage;
    }

    private Mat WarpAndMergeFace(Mat targetImage, Mat alignedImage, Mat normalizationMatrix, Mat fakeImage)
    {
        // Convert to float and compute the average difference between fake and aligned images
        var bgrFakeFloat = ConvertToFloatImage(fakeImage.Clone());
        var alignedImageFloat = ConvertToFloatImage(alignedImage);
        var fakeDiff = ComputeAverageColorDifference(bgrFakeFloat, alignedImageFloat);

        // Zero out the borders of fakeDiff
        ZeroOutImageBorders(fakeDiff);

        // Invert the normalization transform and warp images back to target space
        var inversionMatrix = InvertNormalizationMatrix(normalizationMatrix);
        var whiteImage = CreateConstantColorImage(alignedImage.Rows, alignedImage.Cols, 255.0f);
        var bgrFakeWarped = WarpImageToTarget(bgrFakeFloat, inversionMatrix, targetImage.Size());
        var whiteImageWarped = WarpImageToTarget(whiteImage, inversionMatrix, targetImage.Size());
        var fakeDiffWarped = WarpImageToTarget(fakeDiff, inversionMatrix, targetImage.Size());

        // Create a mask from the warped white image and threshold the fakeDiffWarped
        var imageMask = CreateMaskFromWhiteImage(whiteImageWarped, threshold: 20);
        ApplyThreshold(fakeDiffWarped, thresholdValue: 10);

        // Find points in the mask and compute mask size for further processing
        var points = GetNonZeroPoints(imageMask);
        var maskSize = CalculateMaskSize(points);

        // Refine the mask (erode it) and apply morphological operations on fakeDiff
        RefineMask(imageMask, maskSize);
        RefineFakeDiff(fakeDiffWarped, maskSize);

        // Convert mask and fakeDiff to normalized float [0,1] range
        NormalizeMaskAndDiff(imageMask, fakeDiffWarped);

        // Merge warped fake image with target image using the mask
        var mergedImage = MergeWarpedWithTarget(bgrFakeWarped, targetImage, imageMask);

        return mergedImage;
    }

    Mat ConvertToFloatImage(Mat image)
    {
        var floatImage = new Mat();
        image.ConvertTo(floatImage, MatType.CV_32F);
        return floatImage;
    }

    Mat ComputeAverageColorDifference(Mat bgrFakeFloat, Mat alignedImageFloat)
    {
        var fakeDiff = new Mat();
        Cv2.Subtract(bgrFakeFloat, alignedImageFloat, fakeDiff);
        fakeDiff = Cv2.Abs(fakeDiff);

        Cv2.Split(fakeDiff, out var diffChannels);
        var averaged = (diffChannels[0] + diffChannels[1] + diffChannels[2]) / 3.0;
        return averaged;
    }

    void ZeroOutImageBorders(Mat image)
    {
        int height = image.Rows;
        int width = image.Cols;

        // Zero out top, bottom, left, right borders
        image.RowRange(0, 2).SetTo(0);
        image.RowRange(height - 2, height).SetTo(0);
        image.ColRange(0, 2).SetTo(0);
        image.ColRange(width - 2, width).SetTo(0);
    }

    Mat InvertNormalizationMatrix(Mat normalizationMatrix)
    {
        var inversionMatrix = new Mat();
        Cv2.InvertAffineTransform(normalizationMatrix, inversionMatrix);
        return inversionMatrix;
    }

    Mat CreateConstantColorImage(int rows, int cols, float value)
    {
        return new Mat(rows, cols, MatType.CV_32F, new Scalar(value));
    }

    Mat WarpImageToTarget(Mat sourceImage, Mat transform, Size targetSize)
    {
        var warped = new Mat();
        Cv2.WarpAffine(sourceImage, warped, transform, targetSize, borderValue: Scalar.All(0));
        return warped;
    }

    Mat CreateMaskFromWhiteImage(Mat whiteImageWarped, double threshold)
    {
        var mask = new Mat();
        Cv2.Compare(whiteImageWarped, new Scalar(threshold), mask, CmpType.GT);
        whiteImageWarped.SetTo(new Scalar(255), mask);
        return whiteImageWarped;
    }

    void ApplyThreshold(Mat image, double thresholdValue)
    {
        Cv2.Threshold(image, image, thresholdValue, 255, ThresholdTypes.Binary);
    }

    OpenCvSharp.Point[] GetNonZeroPoints(Mat imageMask)
    {
        var index = new Mat();
        Cv2.FindNonZero(imageMask, index);

        var points = new OpenCvSharp.Point[index.Rows];
        for (int i = 0; i < index.Rows; i++)
        {
            var vec = index.Get<Vec2i>(i);
            points[i] = new Point(vec.Item0, vec.Item1);
        }
        return points;
    }

    int CalculateMaskSize(OpenCvSharp.Point[] points)
    {
        var maskHIndices = points.Select(p => p.Y).ToArray();
        var maskWIndices = points.Select(p => p.X).ToArray();

        var maskHeight = maskHIndices.Max() - maskHIndices.Min();
        var maskWidth = maskWIndices.Max() - maskWIndices.Min();

        int maskSize = (int)Math.Sqrt(maskHeight * maskWidth);
        return maskSize;
    }

    void RefineMask(Mat imageMask, int maskSize)
    {
        // Erode the mask
        var kernelSize = Math.Max(maskSize / 10, 10);
        var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(kernelSize, kernelSize));
        Cv2.Erode(imageMask, imageMask, kernel, iterations: 1);
    }

    void RefineFakeDiff(Mat fakeDiff, int maskSize)
    {
        // Dilate the fakeDiff
        var smallKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2, 2));
        Cv2.Dilate(fakeDiff, fakeDiff, smallKernel, iterations: 1);

        // Apply Gaussian blur with dynamically calculated size
        var blurKernelSize = Math.Max(maskSize / 20, 5);
        var blurSize = new Size(2 * blurKernelSize + 1, 2 * blurKernelSize + 1);
        Cv2.GaussianBlur(fakeDiff, fakeDiff, blurSize, 0);
    }

    void NormalizeMaskAndDiff(Mat imageMask, Mat fakeDiff)
    {
        // Further blur the imageMask for smooth transitions
        var blurKernelSize = 5;
        var blurSize = new Size(2 * blurKernelSize + 1, 2 * blurKernelSize + 1);
        Cv2.GaussianBlur(imageMask, imageMask, blurSize, 0);

        // Normalize to [0,1]
        imageMask.ConvertTo(imageMask, MatType.CV_32F, 1.0 / 255.0);
        fakeDiff.ConvertTo(fakeDiff, MatType.CV_32F, 1.0 / 255.0);
    }

    Mat MergeWarpedWithTarget(Mat bgrFakeWarped, Mat targetImage, Mat imageMask)
    {
        var imageMask3Channels = new Mat();
        Cv2.Merge(new[] { imageMask, imageMask, imageMask }, imageMask3Channels);

        var invertedImageMask = new Mat();
        Cv2.Subtract(Scalar.All(1.0), imageMask3Channels, invertedImageMask);

        var targetImageFloat = ConvertToFloatImage(targetImage);
        var bgrFakeWarpedFloat = ConvertToFloatImage(bgrFakeWarped);

        var temp1 = new Mat();
        Cv2.Multiply(bgrFakeWarpedFloat, imageMask3Channels, temp1);

        var temp2 = new Mat();
        Cv2.Multiply(targetImageFloat, invertedImageMask, temp2);

        var merged = new Mat();
        Cv2.Add(temp1, temp2, merged);

        merged.ConvertTo(merged, MatType.CV_8U);
        return merged;
    }
}