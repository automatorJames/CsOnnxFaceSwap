namespace FaceSwapOnnx;

public enum OnnxModel
{
    RetinaFace,
    Landmark2d,
    Landmark3d,
    GenderAge,
    Arcface,
    InSwapper,
    EsrGan
}

public enum AnalysisPackage
{
    Full,
    Essential,
    SwapOnly
}

public enum Gender
{
    Feminine,
    Masculine,
}

public enum MultiFacePolicy
{
    FirstSourceToAllTargets,
    LastSourceToAllTargets,
    AlternateSourcesToTargets,
    FitMaxSourcesToTargets,
}

public enum BinaryMatrix
{
    EMap,
    MeanShape
}

public enum JobType
{
    FaceSwap,
    SingleFramePreview,
    FaceValidation,
    Caption
}

public enum FauxJobStage
{
    AnalyzeFaces,
    ExtractFrames,
    UpscaleFace,
    FaceSwap
}

public enum JobResultType
{
    None,
    FaceValidationStatus,
    FacePreviewFrame,
    FaceAnalysisFrame,
    FaceSwapInProgressFrame,
    FaceSwappedVideo,
    CaptionedVideo
}

public enum JobStage
{
    Received,
    LoadFace,
    ExportFrames,
    SendFaceAnalysis,
    FaceSwap,
    UpscaleFace,
    CompileVideo,
    Complete
}