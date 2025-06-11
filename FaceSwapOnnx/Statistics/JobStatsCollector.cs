namespace FaceSwapOnnx.Statistics;

public class JobStatsCollector
{
    Dictionary<Guid, JobStats> _jobs = new();

    public void SignalJobStart(Guid jobId, Guid itemId, Type onnxModelType) => SignalJobStatus(jobId, itemId, onnxModelType, isEnd: false);

    public void SignalJobEnd(Guid jobId, Guid itemId, Type onnxModelType) => SignalJobStatus(jobId, itemId, onnxModelType, isEnd: true);

    public void SignalJobStatus(Guid jobId, Guid itemId, Type callingType, bool isEnd)
    {
        if (!_jobs.ContainsKey(jobId))
            _jobs[jobId] = new JobStats(jobId);

        var signalTime = DateTime.Now;
        var job = _jobs[jobId];

        if (isEnd)
            job.EndTimes[(itemId, callingType)] = signalTime;
        else
            job.StartTimes[(itemId, callingType)] = signalTime;
    }

    public void PrintAllJobStats()
    {
        foreach (var job in _jobs.Values)
            Console.WriteLine(job.GetDurationForAllTypesString());
    }

    public void PrintLastJobStats()
    {
        if (_jobs.Any())
            Console.WriteLine(Environment.NewLine + _jobs.Last().Value.GetDurationForAllTypesString());
    }

}
