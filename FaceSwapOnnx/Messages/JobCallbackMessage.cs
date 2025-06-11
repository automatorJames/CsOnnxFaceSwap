namespace FaceSwapOnnx.Messages;

public class JobCallbackMessage
{
    public Guid JobId { get; set; }

    /// <summary>
    /// To be used generically identify external callback systems or processes 
    /// Example: SignalR Hub ConnectionId 
    /// </summary>
    public string ExternalSystemId { get; set; }
    //public JobStage? Stage { get; set; }
    public JobResultType ResultType { get; set; }
    public byte[] ResultByteArray { get; set; }
    public string ResultFilename { get; set; }
    public double PercentComplete { get; set; }
    public int ResultFrameIndex { get; set; }
    public bool IsSuccess { get; set; } = true;

    public JobCallbackMessage()
    {
    }

    public JobCallbackMessage(FaceSwapJob job)
    {
        JobId = job.JobId;
        ExternalSystemId = job.ExternalSystemId;
    }

    public JobCallbackMessage(FaceSwapJob job, JobResultType resultType, byte[] result, string resultFilename = null, double percentComplete = 0, int resultFrameIndex = 0, bool isSuccess = true)
    {
        JobId = job.JobId;
        ExternalSystemId = job.ExternalSystemId;
        ResultType = resultType;
        ResultByteArray = result;
        PercentComplete = percentComplete;
        ResultFrameIndex = resultFrameIndex;
        ResultFilename = resultFilename;
        IsSuccess = isSuccess;
    }

    public JobCallbackMessage(FaceSwapJob job, JobResultType resultType, bool isSuccess)
    {
        JobId = job.JobId;
        ExternalSystemId = job.ExternalSystemId;
        ResultType = resultType;
        IsSuccess = isSuccess;
    }



    //public static JobCallbackMessage GetJobStageMessage(FaceSwapJob job, JobStage jobStage) => new JobCallbackMessage(job) { Stage = jobStage };
}
