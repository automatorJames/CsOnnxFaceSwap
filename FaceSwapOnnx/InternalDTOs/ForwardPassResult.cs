namespace FaceSwapOnnx.InternalDTOs;

public class ForwardPassResult
{
    public List<Mat> Scores { get; set; } = new();
    public List<Mat> BoundingBoxes { get; set; } = new();
    public List<Mat> KeyPointSets { get; set; } = new();
}
