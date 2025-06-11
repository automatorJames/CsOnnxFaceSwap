namespace FaceSwapOnnx;

public class FaceSwapperOptions
{
    public bool UseGpu { get; set; }
    public bool WarmupInferenceAtStartup { get; set; }
    public bool TrackStatistics { get; set; }
    public MultiFacePolicy DefaultMultiFacePolicy { get; set; }
    public string RetinaOnnxFilePath { get; set; }
    public string Landmark3DOnnxFilePath { get; set; }
    public string Landmark2DOnnxFilePath { get; set; }
    public string GenderAgeOnnxFilePath { get; set; }
    public string ArcFaceOnnxFilePath { get; set; }
    public string INSwapperOnnxFilePath { get; set; }
    public string EsrGanOnnxFilePath { get; set; }

    // Environment-specific directories
    public string LocalImageDir { get; set; }
    public string LocalVideoDir { get; set; }
    public string LocalTempDir { get; set; }

}
