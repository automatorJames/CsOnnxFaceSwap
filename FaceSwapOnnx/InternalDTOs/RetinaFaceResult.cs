namespace FaceSwapOnnx.InternalDTOs;

public class RetinaFaceResult
{
    public Guid JobId { get; set; }
    public Guid FaceId { get; set; }
    public Mat BoundingBoxesAndDetectionScores { get; set; }
    public Mat FaceDetectionKeyPoints { get; set; }
    public Mat Image { get; set; }

    public RetinaFaceResult(Guid jobId, Guid faceId, Mat detectionScores, Mat keyPointsSet, Mat image)
    {
        JobId = jobId;
        FaceId = faceId;
        BoundingBoxesAndDetectionScores = detectionScores;
        FaceDetectionKeyPoints = keyPointsSet;
        Image = image;
    }
}
