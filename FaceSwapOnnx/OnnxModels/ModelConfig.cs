namespace FaceSwapOnnx.OnnxModels;

public class ModelConfig
{
    public OnnxModel Model { get; set; }
    public string OnnxFilePath { get; set; }
    public double InputMean { get; set; }
    public double InputStandard { get; set; }
    public int[] InputShape { get; set; }
    public Tuple<int, int> InputSize { get; set; }
    public List<string> InputNames { get; set; }
    public FaceSwapperOptions Options { get; set; }
    public InferenceSession Session { get; set; }

    public ModelConfig(
        OnnxModel model, 
        FaceSwapperOptions swapshuns,
        double inputMean,
        double inputStandard,
        int? inputSizeOverride = null,
        List<Tuple<int, int[]>>? shapeOverridesAtInputIndex = null
        )
    {
        Options = swapshuns;

        var onnxFilePath = model switch
        {
            OnnxModel.Arcface => swapshuns.ArcFaceOnnxFilePath, 
            OnnxModel.GenderAge => swapshuns.GenderAgeOnnxFilePath, 
            OnnxModel.InSwapper => swapshuns.INSwapperOnnxFilePath, 
            OnnxModel.Landmark2d => swapshuns.Landmark2DOnnxFilePath, 
            OnnxModel.Landmark3d => swapshuns.Landmark3DOnnxFilePath, 
            OnnxModel.RetinaFace => swapshuns.RetinaOnnxFilePath, 
            OnnxModel.EsrGan => swapshuns.EsrGanOnnxFilePath, 
            _ => string.Empty
        };

        var sessionOptions = new SessionOptions();

        if (Options.UseGpu)
        {
            sessionOptions.AppendExecutionProvider_CUDA(deviceId: 0);
            Session = new InferenceSession(onnxFilePath, sessionOptions);
        }
        else
        {
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            Session = new InferenceSession(onnxFilePath, sessionOptions);
        }

        InputMean = inputMean;
        InputStandard = inputStandard;

        Model = model;
        OnnxFilePath = onnxFilePath;

        var inputs = Session.InputMetadata;
        InputNames = inputs.Keys.ToList();

        var firstInputInfo = inputs[InputNames.First()];
        InputShape = firstInputInfo.Dimensions.ToArray();

        var inputHeight = inputSizeOverride ?? InputShape[3];
        var inputWidth = inputSizeOverride ?? InputShape[2];

        InputSize = new Tuple<int, int>(inputHeight, inputWidth); // Assuming NCHW format
    }
}
