using FaceSwapOnnx.Statistics;

namespace FaceSwapOnnx.Models;

public class Landmark2d : LandmarkBase
{
    public Landmark2d(FaceSwapperOptions options, JobStatsCollector jobStatsCollector) : base(
        options: options,
        jobStatsCollector,
        OnnxModel.Landmark2d,
        landmarkDimensions: 2,
        landmarkCount: 106,
        requirePose: false) {}
}