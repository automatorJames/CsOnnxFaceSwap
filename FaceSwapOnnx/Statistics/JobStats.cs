namespace FaceSwapOnnx.Statistics;

public class JobStats(Guid jobId)
{
    public Guid JobKey { get; set; } = jobId;

    public Dictionary<(Guid itemId, Type callingType), DateTime> StartTimes { get; set; } = new();
    public Dictionary<(Guid itemId, Type callingType), DateTime> EndTimes { get; set; } = new();

    TimeSpan GetDurationTotal()
    {
        if (!StartTimes.Any() || !EndTimes.Any())
            return TimeSpan.Zero;

        return EndTimes.Max(x => x.Value) - StartTimes.Min(x => x.Value);
    }

    TimeSpan GetDurationForType(Type type)
    {
        if (!StartTimes.Any(x => x.Key.callingType == type) || !StartTimes.Any(x => x.Key.callingType == type))
            return TimeSpan.Zero;

        var startItemsOfType = StartTimes.Where(x => x.Key.callingType.Equals(type));
        var endItemsOfType = EndTimes.Where(x => x.Key.callingType.Equals(type));
        var sharedItemIds = startItemsOfType.Select(x => x.Key.itemId).Intersect(endItemsOfType.Select(x => x.Key.itemId));

        List<TimeSpan> durations = new();
        foreach (var itemId in sharedItemIds)
        {
            var startTime = startItemsOfType.First(x => x.Key.itemId.Equals(itemId)).Value;
            var endTime = endItemsOfType.First(x => x.Key.itemId.Equals(itemId)).Value;
            durations.Add(endTime - startTime);
        }

        return durations.Aggregate(TimeSpan.Zero, (x, sum) => sum + x);
    }

    public (Type type, TimeSpan duration)[] GetDurationForAllTypes()
    {
        List<(Type, TimeSpan)> list = new();

        foreach (var type in StartTimes.Keys.Select(x => x.callingType).Where(x => EndTimes.Keys.Select(y => y.callingType).Contains(x)))
            list.Add((type, GetDurationForType(type)));

        return list.ToArray();
    }

    public string GetDurationTotalString() => GetDurationTotal().Humanize();

    public string GetDurationForTypeString(Type type) => GetDurationForType(type).Humanize();

    public string GetDurationForAllTypesString()
    {
        var durationForAllTypes = GetDurationForAllTypes();
        var totalDuration = durationForAllTypes.Aggregate(TimeSpan.Zero, (sum, x) => sum + x.duration);
        var allTypesString = string.Join(Environment.NewLine, durationForAllTypes.Select(x => $"{x.type.Name, -16}: {x.duration.Humanize()}"));
        allTypesString += $"{Environment.NewLine}{Environment.NewLine}{"Total", -16}: {totalDuration.Humanize()}";
        return allTypesString;
    }
        
}
