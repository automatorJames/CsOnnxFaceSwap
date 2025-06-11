namespace FaceSwapOnnx.Messages;

public class JobUpdateCallbacks
{
    public Func<JobStage, Task> OnJobStageUpdate { get; set; }
    public Func<string, Task> OnFaceAnalysisFrameComplete { get; set; }
    public Func<(int frameIndex, string imagePath), Task> OnFrameSwapComplete { get; set; }
    public Func<double, Task> OnFrameSwapeCompletionUpdate { get; set; }
    public Func<string, Task> OnSwappedVideoComplete { get; set; }

    //public async Task TryOnJobStageUpdate(FaceSwapJobProgress progress)
    //{
    //    if (OnJobStageUpdate is not null)
    //        await OnJobStageUpdate(progress);
    //}
    //
    //public async Task TryOnFaceAnalysisFrameComplete(string faceAnalysisFramePath)
    //{
    //    if (OnFaceAnalysisFrameComplete is not null)
    //        await OnFaceAnalysisFrameComplete(faceAnalysisFramePath);
    //}
    //
    //public async Task TryOnFrameSwapComplete(int frameIndex, string imagePath)
    //{
    //    if (OnFrameSwapComplete is not null)
    //        await OnFrameSwapComplete((frameIndex, imagePath));
    //}
    //
    //public async Task TryOnFrameSwapeCompletionUpdate(double completionPercent)
    //{
    //    if (OnFrameSwapeCompletionUpdate is not null)
    //        await OnFrameSwapeCompletionUpdate(completionPercent);
    //}

    public async Task TryOnSwappedVideoComplete(string videoPath)
    {
        if (OnSwappedVideoComplete is not null)
            await OnSwappedVideoComplete(videoPath);
    }
}
