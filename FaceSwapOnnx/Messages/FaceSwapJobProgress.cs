namespace FaceSwapOnnx;

public class FaceSwapJobProgress
{
    public DateTime JobReceivedTime { get; set; }
    public DateTime? AnalyzeSourceFaceStartTime { get; set; }
    public DateTime? ExtractFramesStartTime { get; set; }
    public DateTime? FaceSwapStartTime { get; set; }
    public DateTime? UpscaleFaceStartTime { get; set; }
    public DateTime? CompileVideoStartTime { get; set; }
    public DateTime? CompleteTime { get; set; }
    public JobStage JobStage { get; set; }

    public int ExtractedFrameCount { get; set; }
    public int FaceSwapFramesProcessed { get; set; }

    [JsonIgnore]
    public double FaceSwapFramesPctComplete { get => CalculateFrameCountPercentageComplete(nameof(FaceSwapFramesProcessed), nameof(FaceSwapStartTime)); }
    public int UpscaleFaceFramesProcessed { get; set; }

    [JsonIgnore]
    public double UpscaleFaceFramesPctComplete { get => CalculateFrameCountPercentageComplete(nameof(UpscaleFaceFramesProcessed), nameof(UpscaleFaceStartTime)); }
    public int EnhancedFramesProcessed { get; set; }

    public TimeSpan CalculateRemainingFaceSwapTime()
    {
        if (ExtractedFrameCount == 0 || FaceSwapStartTime is null)
            return TimeSpan.Zero;

        var timeElapsedSinceSectionStart = DateTime.UtcNow - FaceSwapStartTime.Value;
        var timePerFrame = timeElapsedSinceSectionStart / FaceSwapFramesProcessed;
        var framesRemaining = ExtractedFrameCount - FaceSwapFramesProcessed;
        return timePerFrame * framesRemaining;
    }

    public double EstimatePctComplete(string startTimePropName, JobStage associatedJobStage, int percentToAddPerSecond = 12, int startingPercent = 33)
    {
        var startTimePropVal = (DateTime?)GetType().GetProperty(startTimePropName).GetValue(this);
        
        if (startTimePropVal is null)
            return 0;
        
        if (JobStage == associatedJobStage)
        {
            var secondsSinceStart = (DateTime.UtcNow - startTimePropVal.Value).TotalSeconds;
            double estimate = startingPercent + secondsSinceStart * percentToAddPerSecond;
            return Math.Min(estimate, 100);
        }
        
        else if (JobStage > associatedJobStage)
            return 100;
        
        return 0;
    }

    public double CalculateFrameCountPercentageComplete(string countPropName, string startTimePropName)
    {
        var completedClipCountPropVal = (int)GetType().GetProperty(countPropName).GetValue(this);
        var startTimePropVal = (DateTime?)GetType().GetProperty(startTimePropName).GetValue(this);

        if (ExtractedFrameCount == 0 || startTimePropVal is null)
            return 0;

        return completedClipCountPropVal / (double)ExtractedFrameCount * 100;
    }
}
