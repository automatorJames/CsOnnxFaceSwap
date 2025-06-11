using FaceSwapOnnx.Statistics;

namespace FaceSwapOnnx.Models;

public class Landmark3d : LandmarkBase
{
    public Landmark3d(FaceSwapperOptions options, JobStatsCollector jobStatsCollector) : base(
        options: options,
        jobStatsCollector,
        OnnxModel.Landmark3d,
        landmarkDimensions: 3,
        landmarkCount: 68,
        requirePose: true) {}
}
