namespace FaceSwapOnnx.Models;

public class RetinaFace(FaceSwapperOptions options, JobStatsCollector jobStatsCollector) : OnnxModelBase(GetModelConfig(options), jobStatsCollector)
{
    static ModelConfig GetModelConfig(FaceSwapperOptions options) => new ModelConfig(
            OnnxModel.RetinaFace,
            options,
            inputMean: 127.5,
            inputStandard: 128,
            inputSizeOverride: 640
        );

    private float nonMaxSuppressionThreshold = 0.4f;
    private float detectionThreshold = 0.5f;
    private int featureMapChannels = 3;
    private int[] downsamplingFactors = [8, 16, 32];

    public RetinaFaceResult Detect(Mat image, Guid? jobId = null)
    {
        jobId ??= Guid.NewGuid();
        var faceId = Guid.NewGuid();

        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobStart(jobId.Value, faceId, GetType());

        double imageAspectRatio = (double)image.Height / image.Width;
        int newImageHeight;
        int newImageWidth;

        if (imageAspectRatio > 1)
        {
            newImageHeight = _modelConfig.InputSize.Item2;
            newImageWidth = (int)(newImageHeight / imageAspectRatio);
        }
        else
        {
            newImageWidth = _modelConfig.InputSize.Item1;
            newImageHeight = (int)(newImageWidth * imageAspectRatio);
        }

        double detectionScale = (double)newImageHeight / image.Height;

        Mat resizedImage = new Mat();
        Cv2.Resize(image, resizedImage, new Size(newImageWidth, newImageHeight));

        Mat detectionImage = new Mat(_modelConfig.InputSize.Item1, _modelConfig.InputSize.Item2, MatType.CV_8UC3, new Scalar(0, 0, 0));

        detectionImage.OverlayAtTopLeftCorner(resizedImage);

        var forwardPassResult = ForwardPass(detectionImage, detectionThreshold);

        var scores = forwardPassResult.Scores.VStack();
        var scoresFlattened = forwardPassResult.Scores.MatsToFlat();
        var scoreOrder = scoresFlattened.Select((score, index) => new { score, index }).OrderByDescending(x => x.score).Select(x => x.index).ToArray();

        var boundingBoxes = forwardPassResult.BoundingBoxes.VStack();

        // No faces were detected, return null
        if (boundingBoxes.Rows == 0)
            return null;

        boundingBoxes /= detectionScale;

        var keyPointSets = forwardPassResult.KeyPointSets.VStack();
        keyPointSets /= detectionScale;

        var preliminaryDetectedBoundingBoxes = boundingBoxes.AppendAdditionalChannels(scores);
        preliminaryDetectedBoundingBoxes = preliminaryDetectedBoundingBoxes.ReorderMat(scoreOrder);

        var indicesOfBoundingBoxesToKeep = NonMaxSuppression(preliminaryDetectedBoundingBoxes, nonMaxSuppressionThreshold, scoreOrder);
        var detectedBoundingBoxes = ExtractRowsByIndices(preliminaryDetectedBoundingBoxes, indicesOfBoundingBoxesToKeep);
        keyPointSets = keyPointSets.ReorderMat(scoreOrder);
        keyPointSets = ExtractRowsByIndices(keyPointSets, indicesOfBoundingBoxesToKeep);

        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobEnd(jobId.Value, faceId, GetType());

        return new RetinaFaceResult(jobId.Value, faceId, detectedBoundingBoxes, keyPointSets, image);
    }

    public ForwardPassResult ForwardPass(Mat image, float threshold)
    {
        ForwardPassResult result = new();

        Mat blob = CvDnn.BlobFromImage(
            image: image,
            scaleFactor: 1.0 / _modelConfig.InputStandard,
            size: new Size(image.Width, image.Height),
            mean: new Scalar(_modelConfig.InputMean, _modelConfig.InputMean, _modelConfig.InputMean),
            swapRB: true,
            crop: false);

        var inputTensor = blob.ToDenseTensor();
        var modelInputValues = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_modelConfig.InputNames.First(), inputTensor) };
        var networkOutputs = _modelConfig.Session.Run(modelInputValues).ToList();

        var modelInputHeight = blob.Size(2);
        var modelInputWidth = blob.Size(3);

        for (int i = 0; i < downsamplingFactors.Length; i++)
        {
            int downsamplingFactor = downsamplingFactors[i];

            var scores = networkOutputs[i].ToChanneledMat();
            var boundingBoxPredictions = networkOutputs[i + featureMapChannels].ToChanneledMat();

            boundingBoxPredictions *= downsamplingFactor;

            Mat keyPointSetPredictions = networkOutputs[i + featureMapChannels * 2].ToChanneledMat() * downsamplingFactor;

            int anchorCentersHeight = (int)Math.Ceiling((float)modelInputHeight / downsamplingFactor);
            int anchorCentersWidth = (int)Math.Ceiling((float)modelInputWidth / downsamplingFactor);

            Mat anchorCenters = GenerateMeshGrid(anchorCentersHeight, anchorCentersWidth, downsamplingFactor, newShape: 2);

            int[] positiveScoreIndices = GetPositiveIndices(scores, threshold);
            var boundingBoxes = DistanceToBoundingBoxes(anchorCenters, boundingBoxPredictions);
            var passingScores = ExtractRowsByIndices(scores, positiveScoreIndices);
            var passingBoundingBoxes = ExtractRowsByIndices(boundingBoxes, positiveScoreIndices);

            result.Scores.Add(passingScores);
            result.BoundingBoxes.Add(passingBoundingBoxes);

            var keyPointSets = DistanceToKeyPointSets(anchorCenters, keyPointSetPredictions);
            var passingKeyPointSets = ExtractRowsByIndices(keyPointSets, positiveScoreIndices);
            result.KeyPointSets.Add(passingKeyPointSets);
        }

        return result;
    }

    public static Mat GenerateMeshGrid(int height, int width, int downsamplingFactor, int newShape)
    {
        // Create X and Y coordinate matrices
        Mat xGrid = new Mat(height, width, MatType.CV_32F);
        Mat yGrid = new Mat(height, width, MatType.CV_32F);

        // Populate the grids
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                // we convert int to floats b/c later on we need to do float-on-float math
                // and Cv2 doesn't allow this type of math on unlike types
                xGrid.Set(i, j, j * 1.0f); // Column indices (X-axis)
                yGrid.Set(i, j, i * 1.0f); // Row indices (Y-axis)
            }
        }

        // Reverse the order if specified
        Mat[] channels = [xGrid, yGrid];

        Mat meshGrid = new Mat();
        Cv2.Merge(channels, meshGrid);

        return meshGrid
            .ScaleAndReshape(downsamplingFactor)
            .ExpandAndReshape(newShape);
    }

    public int[] GetPositiveIndices(Mat scores, float threshold)
    {
        // Validate that scores is a single-channel float matrix
        if (scores.Type() != MatType.CV_32FC1)
            throw new ArgumentException("Scores must be of type CV_32FC1.");

        // Create a binary mask where scores >= threshold
        Mat mask = new Mat();
        Cv2.Compare(scores, threshold, mask, CmpType.GE);

        // Find the indices where the mask is non-zero
        Mat nonZeroLocations = new Mat();
        Cv2.FindNonZero(mask, nonZeroLocations);

        // Extract the indices
        int[] positiveIndices = new int[nonZeroLocations.Rows];

        for (int i = 0; i < nonZeroLocations.Rows; i++)
            positiveIndices[i] = nonZeroLocations.At<Point>(i).Y; // Y coordinate is the row index

        return positiveIndices;
    }

    public Mat ExtractRowsByIndices(Mat mat, int[] indices)
    {
        // Validate that mat is not empty
        if (mat.Empty())
            throw new ArgumentException("Input Mat is empty.");

        // Create an output Mat to hold the extracted rows
        Mat extracted = new Mat(indices.Length, mat.Cols, mat.Type());

        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indices[i];

            // Ensure index is within bounds
            if (idx < 0 || idx >= mat.Rows)
                throw new IndexOutOfRangeException($"Index {idx} is out of bounds for Mat with {mat.Rows} rows.");

            // Copy the row at idx to the extracted Mat
            mat.Row(idx).CopyTo(extracted.Row(i));
        }

        return extracted;
    }

    public static Mat DistanceToBoundingBoxes(Mat points, Mat distances)
    {
        if (points.Channels() != 2 || distances.Channels() != 4)
            throw new ArgumentException("Points must have 2 channels and distances must have 4 channels.");

        // Split the input Mats into individual channels
        Mat[] pointChannels = Cv2.Split(points); // [x, y]
        Mat[] distanceChannels = Cv2.Split(distances); // [left, top, right, bottom]

        // Compute bounding box coordinates
        Mat x1 = pointChannels[0] - distanceChannels[0];
        Mat y1 = pointChannels[1] - distanceChannels[1];
        Mat x2 = pointChannels[0] + distanceChannels[2];
        Mat y2 = pointChannels[1] + distanceChannels[3];

        // Merge results into a single Mat
        Mat[] boundingBoxChannels = { x1, y1, x2, y2 };
        Mat boundingBoxes = new Mat();
        Cv2.Merge(boundingBoxChannels, boundingBoxes);

        return boundingBoxes;
    }

    public static Mat DistanceToKeyPointSets(Mat points, Mat distances)
    {
        if (points.Channels() != 2)
            throw new ArgumentException("Points must have 2 channels.");
        if (distances.Channels() % 2 != 0)
            throw new ArgumentException("Distances must have an even number of channels.");

        int numKeyPoints = distances.Channels() / 2;
        Mat[] predictions = new Mat[numKeyPoints * 2];

        Mat[] pointChannels = Cv2.Split(points); // [x, y]
        Mat[] distanceChannels = Cv2.Split(distances);

        for (int i = 0, predIdx = 0; i < distances.Channels(); i += 2, predIdx += 2)
        {
            Mat px = pointChannels[0] + distanceChannels[i];
            Mat py = pointChannels[1] + distanceChannels[i + 1];

            predictions[predIdx] = px;
            predictions[predIdx + 1] = py;
        }

        // Merge predictions into a single Mat
        Mat keyPoints = new Mat();
        Cv2.Merge(predictions, keyPoints);

        return keyPoints.Reshape(2, keyPoints.Rows); // Reshape to (N, 2)
    }

    public static int[] NonMaxSuppression(Mat dets, float nmsThresh, int[] orderArray)
    {
        var order = orderArray.ToList();

        var channels = Cv2.Split(dets);
        Mat x1 = channels[0];
        Mat y1 = channels[1];
        Mat x2 = channels[2];
        Mat y2 = channels[3];
        Mat scores = channels[4];

        // Compute areas of bounding boxes
        Mat areas = (x2 - x1 + new Scalar(1)).Mul(y2 - y1 + new Scalar(1));

        List<int> keep = new List<int>();

        while (order.Count > 0)
        {
            int i = order[0]; // Index of the current box with the highest score
            keep.Add(i);
            order.RemoveAt(0);

            if (order.Count == 0)
                break;

            // Extract coordinates of the highest-scoring box
            float x1_i = x1.At<float>(i);
            float y1_i = y1.At<float>(i);
            float x2_i = x2.At<float>(i);
            float y2_i = y2.At<float>(i);
            float area_i = areas.At<float>(i);

            // Extract coordinates and areas of the remaining boxes
            Mat x1_order = x1.ExtractValuesByIndices(order);
            Mat y1_order = y1.ExtractValuesByIndices(order);
            Mat x2_order = x2.ExtractValuesByIndices(order);
            Mat y2_order = y2.ExtractValuesByIndices(order);
            Mat areas_order = areas.ExtractValuesByIndices(order);

            // Compute IoU (intersection over union)
            Mat xx1 = new();
            Mat yy1 = new();
            Mat xx2 = new();
            Mat yy2 = new();

            Cv2.Max(x1_i, x1_order, xx1);
            Cv2.Max(y1_i, y1_order, yy1);
            Cv2.Min(x2_i, x2_order, xx2);
            Cv2.Min(y2_i, y2_order, yy2);

            Mat w = new();
            Mat h = new();
            Cv2.Max(xx2 - xx1 + new Scalar(1), 0, w);
            Cv2.Max(yy2 - yy1 + new Scalar(1), 0, h);

            Mat inter = w.Mul(h);
            Mat ovr = inter / (new Scalar(area_i) + areas_order - inter);

            // Find indices where IoU is below the threshold
            List<int> inds = new List<int>();
            for (int idx = 0; idx < ovr.Rows; idx++)
            {
                if (ovr.At<float>(idx) <= nmsThresh)
                    inds.Add(order[idx]);
            }

            // Update order to keep boxes with IoU below threshold
            order = inds;
        }

        return keep.ToArray();
    }
}
