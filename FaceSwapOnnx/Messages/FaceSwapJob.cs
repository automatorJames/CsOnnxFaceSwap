using Newtonsoft.Json;

namespace FaceSwapOnnx.Messages;

public class FaceSwapJob
{
    public Guid JobId { get; set; } = Guid.NewGuid();

    /// <summary>
    /// To be used generically identify external callback systems or processes 
    /// Example: SignalR Hub ConnectionId 
    /// </summary>
    public string ExternalSystemId { get; set; }
    public List<JobType> JobTypes { get; set; }
    public string Username { get; set; }
    public List<FaceMetadata> FaceFiles { get; set; }
    public bool UseCachedImages { get; set; }
    public string VideoFilename { get; set; }
    public bool UpscaleFace { get; set; }
    public int? HeightLimit { get; set; }
    public int? FpsLimit { get; set; }
    public int? NumberOfClipsDesired { get; set; }
    public int NumberOfClips { get; set; }
    public string VideoInTime { get; set; }
    public string VideoOutDuration { get; set; }
    public string OutputVideoBaseName { get; set; }
    public bool CombineOutput { get; set; }
    public FaceSwapJobProgress JobProgress { get; set; }
    public int? RequestImgPreviewEveryNFrames { get; set; }
    public bool RequestFaceAnalysisImg { get; set; }

    // These props are set by the server after receiving the message
    public string ServerInputVideoPath { get; set; }
    public DirectoryInfo ServerTempContainerDir { get; set; }
    public string ServerOriginalFramesDir => Path.Combine(ServerTempContainerDir?.FullName ?? string.Empty, "original-frames");
    public string ServerSwappedFramesDir => Path.Combine(ServerTempContainerDir?.FullName ?? string.Empty, "swapped-frames");
    public string ServerOutputVideoFilepath { get; set; }
    public int OutputVideoHeight { get; set; }
    public double OutputVideoFps { get; set; }

    public Func<JobCallbackMessage, Task> OnCallback { get; set; }

    /// <summary>
    /// This is set by the service that processes the job message.
    /// At allows the FaceSwapper to communicate with the host service
    /// so that async callbacks can be made (e.g. SignalR updates)
    /// </summary>
    public JobUpdateCallbacks JobUpdateCallbacks { get; set; }

    public FaceSwapJob()
    {
    }

    public void SetServerPaths(FaceSwapperOptions options)
    {
        ServerTempContainerDir = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), JobId.ToString()));
        Directory.CreateDirectory(ServerTempContainerDir.FullName);

        string outputVideoFilename = null;

        if (JobTypes.Contains(JobType.FaceSwap) || JobTypes.Contains(JobType.SingleFramePreview) || JobTypes.Contains(JobType.Caption))
        {
            ServerInputVideoPath = Path.Combine(options.LocalVideoDir, VideoFilename);

            foreach (var facefile in FaceFiles)
                facefile.ServerSideFilePath = Path.Combine(options.LocalImageDir, Username, facefile.FileName);

            outputVideoFilename = $"{OutputVideoBaseName ?? "swapped"}_{string.Join('_', FaceFiles.Select(x => Path.GetFileNameWithoutExtension(x.FileName)))}.mp4";
            Directory.CreateDirectory(ServerOriginalFramesDir);
            Directory.CreateDirectory(ServerSwappedFramesDir);
        }

        if (JobTypes.Contains(JobType.Caption))
            outputVideoFilename = $"{OutputVideoBaseName ?? "captioned"}.mp4";

        if (JobTypes.Contains(JobType.FaceSwap) || JobTypes.Contains(JobType.Caption) || JobTypes.Contains(JobType.SingleFramePreview))
            ServerOutputVideoFilepath = Path.Combine(ServerTempContainerDir.FullName, outputVideoFilename);
    }

    public void DeleteServerTempPath()
    {
        if (ServerTempContainerDir?.FullName is not null && Directory.Exists(ServerTempContainerDir.FullName))
            Directory.Delete(ServerTempContainerDir.FullName, true);
    }

    public void SetServerSideVideoProperties(VideoFileInfo videoFileInfo)
    {
        OutputVideoHeight = HeightLimit ?? videoFileInfo.Height;
        OutputVideoFps = FpsLimit ?? videoFileInfo.Fps;
    }

    public FaceSwapJob(bool isFaceSwap = false, bool isSingleFramePreview = false, bool isFaceValidation = false, bool isCaption = false)
    {
        JobId = Guid.NewGuid();

        JobTypes = new();
        if (isFaceSwap)
            JobTypes.Add(JobType.FaceSwap);
        if (isSingleFramePreview)
            JobTypes.Add(JobType.SingleFramePreview);
        if (isFaceValidation)
            JobTypes.Add(JobType.FaceValidation);
        if (isCaption)
            JobTypes.Add(JobType.Caption);

        FaceFiles = new();
    }

    public FaceSwapJob(FaceSwapJob message)
    {
        JobId = message.JobId;
        ExternalSystemId = message.ExternalSystemId;
        JobTypes = message.JobTypes;
        Username = message.Username;
        FaceFiles = message.FaceFiles;
        UseCachedImages = message.UseCachedImages;
        VideoFilename = message.VideoFilename;
        UpscaleFace = message.UpscaleFace;
        HeightLimit = message.HeightLimit;
        FpsLimit = message.FpsLimit;
        NumberOfClips = message.NumberOfClips;
        NumberOfClipsDesired = message.NumberOfClipsDesired;
        CombineOutput = message.CombineOutput;
        JobProgress = message.JobProgress;
        RequestImgPreviewEveryNFrames = message.RequestImgPreviewEveryNFrames;
    }

    public List<string> GetAllFaceNames() => FaceFiles.Select(x => x.FaceName).ToList();

    public async Task TryCallbackAsync(JobCallbackMessage msg)
    {
        if (OnCallback is not null)
        {
            msg.JobId = JobId;
            await OnCallback(msg);
        }
    }

    public override string ToString()
    {
        return $@"
Face(s          {GetAllFaceNames()}
Clip ID:        {VideoFilename} 
Upscale Face:   {UpscaleFace} 
Height Limit:   {HeightLimit?.ToString() ?? "N/A"}
FPS Limit:      {FpsLimit?.ToString() ?? "N/A"}";
    }

    public string ToJsonDebug() => JsonConvert.SerializeObject(this, Formatting.Indented);
}
