namespace FaceSwapOnnx.OnnxModels;

public class OnnxModelBase
{
    protected ModelConfig _modelConfig;
    protected JobStatsCollector _jobStatsCollector;
    protected InferenceSession _session;
    protected FaceSwapperOptions _options;

    public OnnxModelBase(ModelConfig modelConfig, JobStatsCollector jobStatsCollector)
    {
        _modelConfig = modelConfig;
        _session = modelConfig.Session;
        _jobStatsCollector = jobStatsCollector;
        _options = _modelConfig.Options;
    }

    public void Warmup()
    {
        List<NamedOnnxValue> modelInputValues = new();

        for (int i = 0; i < _modelConfig.InputNames.Count; i++)
        {
            var inputName = _modelConfig.InputNames[i];

            // todo: this is a hack that only works b/c there's only one model with two inputs
            int[] shape = i == 0 ? _modelConfig.InputShape : [1, 512];

            // todo: this is a RetinaFace-only hack
            if (shape.Length == 4 && _modelConfig.Model == OnnxModel.RetinaFace)
            {
                shape[2] = 640;
                shape[3] = 640;
            }

            var dummyData = GetDummyData(shape);
            modelInputValues.Add(NamedOnnxValue.CreateFromTensor(inputName, dummyData));
        }

        _session.Run(modelInputValues);
    }
}
