using FaceSwapOnnx.HelperMathClasses;

namespace FaceSwapOnnx.Models;

public class LandmarkBase(
    FaceSwapperOptions options, 
    JobStatsCollector jobStatsCollector, 
    OnnxModel model,
    int landmarkDimensions, 
    int landmarkCount, 
    bool requirePose) : OnnxModelBase(GetModelConfig(options, model), jobStatsCollector)
{
    static ModelConfig GetModelConfig(FaceSwapperOptions options, OnnxModel model) => new ModelConfig(
            model,
            options,
            inputMean: 0.0,
            inputStandard: 1.0
        );

    protected List<string> _outputNames;
    protected int _landmarkDimensions = landmarkDimensions;
    protected int _landmarkCount = landmarkCount;
    protected bool _requirePose = requirePose;
    protected string _taskName;

    public Mat Process(Mat img, Face face)
    {
        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobStart(face.JobId, face.FaceId, GetType());

        // Extract bbox and compute center, scale, rotate
        double[] boundingBox = face.BoundingBox.PointsFlattened; // [x1, y1, x2, y2]
        double w = boundingBox[2] - boundingBox[0];
        double h = boundingBox[3] - boundingBox[1];
        var center = Tuple.Create((boundingBox[2] + boundingBox[0]) / 2.0f, (boundingBox[3] + boundingBox[1]) / 2.0f);
        double rotate = 0;
        double _scale = _modelConfig.InputSize.Item1 / (Math.Max(w, h) * 1.5f);

        Mat alignedImage, M;
        FaceAlign.Transform(img, center, _modelConfig.InputSize.Item1, _scale, rotate, out alignedImage, out M);

        // Prepare the blob for input
        Mat imageBlob = CvDnn.BlobFromImage(
            alignedImage,
            1.0 / _modelConfig.InputStandard,
            new Size(_modelConfig.InputSize.Item1, _modelConfig.InputSize.Item2),
            new Scalar(_modelConfig.InputMean, _modelConfig.InputMean, _modelConfig.InputMean),
            swapRB: true);

        // Create input tensor using your extension method
        var inputTensor = imageBlob.ToDenseTensor();

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_modelConfig.InputNames[0], inputTensor)
        };

        using var result = _session.Run(inputs).First();
        var pred = result.ToChanneledMat(swapAxes: true, segmentIntoChannelSize: _landmarkDimensions);
        var predTensor = result.AsTensor<float>();

        // If lmk_num < pred.Rows, select the last 'lmk_num' rows
        if (_landmarkCount < pred.Rows)
            pred = pred.RowRange(pred.Rows - _landmarkCount, pred.Rows);

        for (int row = 0; row < pred.Rows; row++)
        {
            Vec3f value = pred.At<Vec3f>(row, 0);
            value.Item0 = (value.Item0 + 1.0f) * (float)Math.Floor(_modelConfig.InputSize.Item1 / 2.0f);
            value.Item1 = (value.Item1 + 1.0f) * (float)Math.Floor(_modelConfig.InputSize.Item1 / 2.0f);

            // If 3D landmarks, adjust the third channel (Channel 2)
            if (_landmarkDimensions == 3)
                value.Item2 = value.Item2 * (float)(_modelConfig.InputSize.Item1 / 2.0f);

            pred.Set(row, 0, value);
        }

        // Invert affine transformation
        Mat IM = new Mat();
        Cv2.InvertAffineTransform(M, IM);

        // Transform points back to original image coordinates
        pred = TransformHelpers.TransPoints(pred, IM);

        // Pose estimation if required
        if (_requirePose)
        {
            var columnWisePred = pred.ConvertChannelsToColumns();
            var meanShape = Static.GetBinaryMatrix(BinaryMatrix.MeanShape);
            Mat P = TransformHelpers.EstimateAffineMatrix3D23D(meanShape, columnWisePred);

            var (s, R, t) = TransformHelpers.P2sRt(P);
            var (rx, ry, rz) = TransformHelpers.Matrix2Angle(R);

            Mat pose = new Mat(1, 3, MatType.CV_32F);
            pose.Set(0, 0, (float)rx);
            pose.Set(0, 1, (float)ry);
            pose.Set(0, 2, (float)rz);

            face.Pose = new FacePose(rx, ry, rz);
        }

        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobEnd(face.JobId, face.FaceId, GetType());

        return pred;
    }
}
