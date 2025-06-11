namespace FaceSwapOnnx.Ffmpeg;

public class FfmpegClient
{
    public string _ffmpegPath;

    public FfmpegClient(string ffmpegPath)
    {
        _ffmpegPath = ffmpegPath;
    }

    public VideoFileInfo GetVideoFileInfo(string inputFilePath)
    {
        var args = $@"-i ""{inputFilePath}"" -hide_banner";
        var result = ExecuteFfmpegWithOutputRedirect(args, hideWindow: true);
        return new VideoFileInfo(result);
    }

    public string ImageSeriesToVideo(string imgDir, string outputFilename, double fps)
    {
        var parentDir = Path.GetDirectoryName(imgDir);
        var outputFilePath = Path.Combine(parentDir, outputFilename);
        var ffmpegCmd = $@" -framerate {fps} -i ""{imgDir}\%04d.png"" -pix_fmt yuv420p ""{outputFilePath}"" -y";
        ExecuteFfmpegWithOutputRedirect(ffmpegCmd);
        return outputFilePath;
    }

    public VideoFileInfo ExportVideoToFrames(string videoPath, string outputDir, string inputTime = null, string duration = null, int? heightLimit = null, int? fpsLimit = null, bool firstFrameOnly = false, string overrideFilename = null, int rotations = 0)
    {
        var videoFileInfo = GetVideoFileInfo(videoPath);
        List<string> vfFilterStatements = new();

        if (heightLimit is not null && videoFileInfo.Height > heightLimit.Value)
        {
            var height = heightLimit.Value;
            var width = (int)Math.Round((double)Convert.ToInt32(height * 1.7777777));
            width += width % 2 != 0 ? 1 : 0;
            vfFilterStatements.Add($"scale={width}:{height}");
        }

        if (fpsLimit is not null && videoFileInfo.Fps > fpsLimit.Value)
            vfFilterStatements.Add($"fps={fpsLimit.Value}");

        var transposeSnippet = "";

        if (rotations == 1)
            transposeSnippet = $" -vf \"transpose=1\"";
        else if (rotations == 2)
            transposeSnippet = $" -vf \"transpose=2,transpose=2\"";
        else if (rotations == 3)
            transposeSnippet = $" -vf \"transpose=2\"";

        var vfFilterSegment = vfFilterStatements.Any() ? "-vf \"" + string.Join(", ", vfFilterStatements) + "\"" : "";
        var frameCountSegment = firstFrameOnly ? " -vframes 1 " : "";
        var fileNameSegment = (firstFrameOnly ? Path.GetFileNameWithoutExtension(videoFileInfo.FilePath) : "%04d") + ".png";
        var hardwareAccelerationSnippet = rotations == 0 ? "-hwaccel cuda " : "";

        // Conditionally include -ss and -to segments
        var inputTimeSegment = !string.IsNullOrEmpty(inputTime) ? $"-ss {inputTime} " : "";
        var toSegment = !string.IsNullOrEmpty(duration) ? $"-to {duration} " : "";

        var filenameToSaveAs = overrideFilename ?? fileNameSegment;

        // Build the ffmpeg command
        var ffmpegCmd = $@"{hardwareAccelerationSnippet}{inputTimeSegment}-i ""{videoFileInfo.FilePath}"" {toSegment}{vfFilterSegment}{frameCountSegment}{transposeSnippet} ""{outputDir}\{filenameToSaveAs}"" -y";
        ExecuteFfmpegWithOutputRedirect(ffmpegCmd);

        return videoFileInfo;
    }

    public void JoinSegments(string concatListPath, string outputFilepath)
    {
        var ffmpegCmd = $@"-f concat -safe 0 -i {concatListPath} -c copy ""{outputFilepath}"" -y";
        ExecuteFfmpegWithOutputRedirect(ffmpegCmd);
    }

    public int GetVideoRotationClockwiseSteps(string filepath)
    {
        var degrees = GetVideoRotationAngle(filepath);

        // compensate for MediaElement interpretting 90 degress backwards
        if (Math.Abs(degrees) == 90)
            degrees += 180;

        int normalized = ((degrees % 360) + 360) % 360;
        return normalized / 90;
    }

    public int GetVideoRotationAngle(string filepath)
    {
        var args = $@"-i ""{filepath}""";
        var result = ExecuteFfmpegWithOutputRedirect(args);
        var match = Regex.Match(result, @"(?<=displaymatrix: rotation of )(-|\d)+(?=\.\d{2})");

        if (match.Success)
            return int.Parse(match.Value);

        return 0;
    }

    public string ExecuteFfmpegWithOutputRedirect(string args, bool hideWindow = false, bool showFfmpegCmdDebug = false)
    {
        using (Process process = GetProcess(args, hideWindow))
        {
            if (showFfmpegCmdDebug)
            {
                Console.WriteLine($@"{Environment.NewLine}""{process.StartInfo.FileName}"" {process.StartInfo.Arguments}");
            }

            // FYI, ffmpeg doesn't actually use StandardOutput -- all output comes from StandardError
            StringBuilder error = new StringBuilder();

            using (var errorWaitHandle = new AutoResetEvent(false))
            {
                process.ErrorDataReceived += (sender, e) =>
                {
                    if (e.Data == null)
                        errorWaitHandle.Set();
                    else
                        error.AppendLine(e.Data);
                };

                process.Start();
                process.BeginErrorReadLine();

                if (process.WaitForExit(300000) && errorWaitHandle.WaitOne(300000))
                {
                    var result = error.ToString();
                    if (process.ExitCode != 0)
                    {
                        // This is an error we expect, because there's no simpler way to get video file info than to not specify an output file
                        if (!result.EndsWith($"At least one output file must be specified{Environment.NewLine}"))
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine("Ffmpeg error");
                            Console.ForegroundColor = ConsoleColor.White;
                            Console.WriteLine(result);
                        }

                    }
                    return result;
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Ffmpeg timeout");
                    Console.ForegroundColor = ConsoleColor.White;
                    return "Timeout";
                }
            }
        }
    }

    public string GetEnableBetweenSnippet(double? fromSecond = null, double? toSecond = null)
    {
        if (fromSecond is null && toSecond is null)
            return string.Empty;

        fromSecond ??= 0;
        toSecond ??= 2;

        return $@"enable='between(t,{fromSecond},{toSecond})'";
    }

    public async Task ExecuteFfmpegAsync(string args, bool hideWindow = true, bool waitForExit = false)
    {
        Debug.WriteLine(args);
        using (Process process = GetProcess(args, hideWindow))
        {
            try
            {
                process.Start();
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.ToString());
            }
        }
    }

    Process GetProcess(string args, bool hideWindow = false)
    {
        var process = new Process();
        process.StartInfo.FileName = _ffmpegPath;
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.Arguments = args;
        return process;
    }

    public TimeSpan GetOffsetStartTime(TimeSpan clipStartTime, int fps, int clipStartFrameOffset = 0, float clipSceneSecondsOffset = 0.0f)
    {
        var offsetStartTime = clipStartTime.Add(TimeSpan.FromSeconds(clipSceneSecondsOffset));

        if (clipStartFrameOffset != 0)
        {
            var fractionalOffset = (1.0 / fps) * clipStartFrameOffset;
            offsetStartTime = offsetStartTime.Add(TimeSpan.FromSeconds(fractionalOffset));
        }

        return offsetStartTime;
    }
}