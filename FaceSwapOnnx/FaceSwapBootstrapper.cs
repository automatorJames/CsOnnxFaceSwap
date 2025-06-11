using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace FaceSwapOnnx;

public static class FaceSwapBootstrapper
{
    public static FaceSwapper GetFaceSwapper(IConfiguration conf)
    {
        var services = new ServiceCollection();
        var options = new FaceSwapperOptions();
        conf.GetSection(nameof(FaceSwapperOptions)).Bind(options);

        AddFaceSwapDependencies(services, conf);

        var serviceProvider = services.BuildServiceProvider();
        return serviceProvider.GetRequiredService<FaceSwapper>();
    }

    public static void AddFaceSwapDependencies(IServiceCollection services, IConfiguration conf)
    {
        var options = new FaceSwapperOptions();
        conf.GetSection(nameof(FaceSwapperOptions)).Bind(options);

        services.AddSingleton(options);
        services.AddSingleton<JobStatsCollector>();
        services.AddSingleton<RetinaFace>();
        services.AddSingleton<Landmark3d>();
        services.AddSingleton<Landmark2d>();
        services.AddSingleton<Arcface>();
        services.AddSingleton<GenderAge>();
        services.AddSingleton<InSwapper>();
        services.AddSingleton<FfmpegClient>();
        services.AddSingleton<FaceSwapper>();
    }

    public static void AddFaceSwapService(this IServiceCollection services, IConfiguration conf) => AddFaceSwapDependencies((IServiceCollection)services, conf);
}
