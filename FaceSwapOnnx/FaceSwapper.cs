using OpenCvSharp;
using System.Drawing;
using System.Drawing.Imaging;
using System.Net.WebSockets;

namespace FaceSwapOnnx;

public class FaceSwapper
{
    public HashSet<string> ImagePathHashes { get; set; } = new();

    public string LocalImageDir { get => _options.LocalImageDir; }
    public string LocalVideoDir { get => _options.LocalVideoDir; }
    public string LocalTempDir { get => _options.LocalTempDir; }

    RetinaFace _retinaFace;
    Landmark3d _landmark3d;
    Landmark2d _landmark2d;
    GenderAge _genderAge;
    Arcface _arcFace;
    InSwapper _inswapper;
    FaceSwapperOptions _options;
    JobStatsCollector _jobStatsCollector;
    FfmpegClient _fmpegClient;
    bool _isInitialized;

    public FaceSwapper(
        FaceSwapperOptions options,
        JobStatsCollector jobStatsCollector,
        RetinaFace retinaFace,
        Landmark3d landmark3d,
        Landmark2d landmark2d,
        GenderAge genderAge,
        Arcface arcFace,
        InSwapper inSwapper,
        FfmpegClient ffmpegClient)
    {
        _options = options;
        _jobStatsCollector = jobStatsCollector;
        _retinaFace = retinaFace;
        _landmark2d = landmark2d;
        _landmark3d = landmark3d;
        _genderAge = genderAge;
        _arcFace = arcFace;
        _inswapper = inSwapper;
        _fmpegClient = ffmpegClient;
    }


    public void Initialize()
    {
        Console.Clear();

        // todo: this strongly couples FaceSwapper with FaceSwapService
        var files = Directory.EnumerateFiles(_options.LocalImageDir, "*", SearchOption.AllDirectories).ToList();
        foreach (var file in files)
        {
            var userDir = Path.GetFileName(Path.GetDirectoryName(file));
            var fileName = Path.GetFileName(file);
            ImagePathHashes.Add(Path.Combine(userDir, fileName));
        }

        if (!_options.WarmupInferenceAtStartup)
            return;

        var stopwatch = Stopwatch.StartNew();
        Console.WriteLine($"Warming up {nameof(RetinaFace)} onnx model...");
        _retinaFace.Warmup();

        Console.WriteLine($"Warming up {nameof(Landmark3d)} onnx model...");
        _landmark3d.Warmup();

        Console.WriteLine($"Warming up {nameof(Landmark2d)} onnx model...");
        _landmark2d.Warmup();

        Console.WriteLine($"Warming up {nameof(GenderAge)} onnx model...");
        _genderAge.Warmup();

        Console.WriteLine($"Warming up {nameof(Arcface)} onnx model...");
        _arcFace.Warmup();

        Console.WriteLine($"Warming up {nameof(InSwapper)} onnx model...");
        _inswapper.Warmup();

        Console.WriteLine($"Warmup complete in {stopwatch.Elapsed.Humanize()}");

        _isInitialized = true;
    }

    public async Task<List<Face>> AnalyzeFacesAsync(string imagePath, AnalysisPackage package = AnalysisPackage.Essential) 
        => await AnalyzeFacesAsync(Cv2.ImRead(imagePath), package);

    public async Task<List<Face>> AnalyzeFacesAsync(Mat image, AnalysisPackage package = AnalysisPackage.Essential)
    {
        var retinaResult = _retinaFace.Detect(image);

        // No faces found, return empty list
        if (retinaResult is null)
            return new List<Face>();

        var faces = Face.FromRetinaFaceResult(retinaResult);

        var modelsToRun = GetAnalysisPackage(package);

        foreach (var face in faces)
        {
            if (modelsToRun.Contains(OnnxModel.Landmark3d))
                await Task.Run(() => _landmark3d.Process(image, face));

            if (modelsToRun.Contains(OnnxModel.Landmark2d))
                await Task.Run(() => _landmark2d.Process(image, face));

            if (modelsToRun.Contains(OnnxModel.GenderAge))
                await Task.Run(() => _genderAge.Process(image, face));

            if (modelsToRun.Contains(OnnxModel.Arcface))
                await Task.Run(() => _arcFace.Process(image, face));
        }

        return faces;
    }

    public string SwapFacesInDir(string sourceImagePath, string targetImagesDir, string outputDir = null, MultiFacePolicy? overridePolicy = null, int parallelism = 6, bool prependFaceName = true)
    {
        outputDir ??= Path.Combine(targetImagesDir, "output");
        var outputDirInfo = Directory.CreateDirectory(outputDir);
        var inputDirInfo = new DirectoryInfo(targetImagesDir);

        var files = inputDirInfo.EnumerateFiles();

        Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, file =>
        {
            var outputFilename = $"{(prependFaceName ? Path.GetFileNameWithoutExtension(sourceImagePath) + "_" : "")}{Path.GetFileNameWithoutExtension(file.Name)}.png";
            var outputFilePath = Path.Combine(outputDir, outputFilename);
            SwapFacesAndSaveOutput(sourceImagePath, file.FullName, outputFilePath, overridePolicy);
        });

        return outputDir;
    }

    public void SwapSingleFace(string sourceImagePath, string targetImagePath)
    {
        var outputDir  = Directory.GetParent(targetImagePath);
        var filename = Path.GetFileNameWithoutExtension(sourceImagePath) + "_" + Path.GetFileName(targetImagePath);
        var outputPath = Path.Combine(outputDir.FullName, filename);
        SwapFacesAndSaveOutput(sourceImagePath, targetImagePath, outputPath);
    }

    public async Task SwapFacesAndSaveOutput(string sourceImagePath, string targetImagePath, string outputPath, MultiFacePolicy? overridePolicy = null)
    {
        Mat matResult = await SwapFacesAsync(sourceImagePath, targetImagePath, overridePolicy);
        var imageResult = BitmapConverter.ToBitmap(matResult, PixelFormat.Format24bppRgb);
        imageResult.Save(outputPath);

        if (_options.TrackStatistics)
            _jobStatsCollector.PrintLastJobStats();
    }

    public async Task<Mat> SwapFacesAsync(string sourceImagePath, string targetImagePath, MultiFacePolicy? overridePolicy = null)
    {
        var sourceFaces = await AnalyzeFacesAsync(sourceImagePath, AnalysisPackage.SwapOnly);
        var targetFaces = await AnalyzeFacesAsync(targetImagePath, AnalysisPackage.SwapOnly);
        var targetImage = Cv2.ImRead(targetImagePath);
        return await SwapFacesAsync(sourceFaces, targetFaces, targetImage, overridePolicy);
    }

    public async Task<Mat> SwapFacesAsync(List<Face> sourceFaces, List<Face> targetFaces, Mat targetImage, MultiFacePolicy? overridePolicy = null)
    {
        if (sourceFaces == null || targetFaces == null)
            throw new ArgumentNullException(nameof(sourceFaces));

        if (!sourceFaces.Any() || !targetFaces.Any())
            throw new ArgumentException("At least one source and one target face are required");

        Mat result = targetImage.Clone();

        // Map as many source faces as possible to target faces
        var count = Math.Min(sourceFaces.Count, targetFaces.Count);
        for (int i = 0; i < count; i++)
        {
            var sourceFace = sourceFaces[i];
            var targetFace = targetFaces[i];
            result = await Task.Run(() => _inswapper.Get(result, targetFace, sourceFace));
        }

        return result;
    }

    public async Task<string> DownscaleImageFor128Face(string imagePath)
    {
        var targetFaces = await AnalyzeFacesAsync(imagePath, AnalysisPackage.SwapOnly);
        var firstFace = targetFaces.First();
        var imageHeight = firstFace.ImageHeight;
        var imageWidth = firstFace.ImageWidth;
        var faceHeight = firstFace.BoundingBox.Rectangle.Height;
        var faceWidth = firstFace.BoundingBox.Rectangle.Width;
        var heightIsGreater = faceHeight >= faceWidth;

        double faceDim = heightIsGreater ? faceHeight : faceWidth;
        double scaleFactor = 128.0 / faceDim;

        int newHeight = (int)(imageHeight * scaleFactor);
        int newWidth = (int)(imageWidth * scaleFactor);

        var originalImage = Cv2.ImRead(imagePath);
        var resizedImage = new Mat();
        Cv2.Resize(originalImage, resizedImage, new OpenCvSharp.Size(newWidth, newHeight));

        var outputPath = Path.Combine(
            Path.GetDirectoryName(imagePath),
            Path.GetFileNameWithoutExtension(imagePath) + "_downscaled128" + Path.GetExtension(imagePath)
        );

        Cv2.ImWrite(outputPath, resizedImage);

        return outputPath;
    }

    public async Task<JobCallbackMessage> ExecuteFaceJobAsync(FaceSwapJob msg)
    {
        var stopwatch = Stopwatch.StartNew();
        msg.SetServerPaths(_options);

        if (msg.JobTypes.Count == 1 && msg.JobTypes.First() == JobType.FaceValidation)
        {
            if (msg.FaceFiles.Count != 1)
                throw new Exception($"Expected one FaceFile, but found {msg.FaceFiles.Count}");

            var faceData = msg.FaceFiles.First().ImgData;

            if (faceData is null)
                throw new Exception($"ImgData cannot be null for FaceValidation jobs");

            var tempFilePath = Path.Combine(msg.ServerTempContainerDir.FullName, msg.FaceFiles.First().FileName);
            File.WriteAllBytes(tempFilePath, faceData);

            var face = await AnalyzeFacesAsync(tempFilePath);
            var success = face.Count == 1;
            var callback = new JobCallbackMessage(msg, JobResultType.FaceValidationStatus, isSuccess: success);
            return callback;
        }
        else
        {

            foreach (var faceFile in msg.FaceFiles)
            {
                if (faceFile.ServerSideFilePath is not null)
                {
                    if (!File.Exists(faceFile.ServerSideFilePath))
                        throw new FileNotFoundException(faceFile.ServerSideFilePath);
                }
                else if (faceFile.ImgData is not null)
                {
                    var tempFilePath = Path.Combine(msg.ServerTempContainerDir.FullName, msg.FaceFiles.First().FileName);
                    File.WriteAllBytes(tempFilePath, faceFile.ImgData);
                    faceFile.ServerSideFilePath = tempFilePath;
                }
                else
                    throw new Exception("ServerSideFilePath and ImgData cannot both be null");

                var facesInCurrentImage = await AnalyzeFacesAsync(faceFile.ServerSideFilePath);

                if (facesInCurrentImage.Count != 1)
                    throw new ArgumentException($"Expected 1 face in {faceFile.ServerSideFilePath}, but found {facesInCurrentImage.Count}");

                faceFile.Face = facesInCurrentImage.First();
            }

            Directory.CreateDirectory(LocalTempDir);

            var targetVideoFileInfo = _fmpegClient.GetVideoFileInfo(msg.ServerInputVideoPath);
            var isSingleFramePreview = msg.JobTypes.Contains(JobType.SingleFramePreview);

            VideoFileInfo framesExportResult = _fmpegClient.ExportVideoToFrames(
                msg.ServerInputVideoPath,
                msg.ServerOriginalFramesDir,
                msg.VideoInTime,
                msg.VideoOutDuration,
                msg.HeightLimit,
                msg.FpsLimit,
                firstFrameOnly: msg.JobTypes.Contains(JobType.SingleFramePreview));

            msg.SetServerSideVideoProperties(framesExportResult);

            var originalFrames = Directory.GetFiles(msg.ServerOriginalFramesDir);
            if (!originalFrames.Any())
                throw new Exception($"No frames found after extraction attempt from {msg.ServerInputVideoPath}");

            if (msg.RequestFaceAnalysisImg)
            {
                var firstFramePath = originalFrames.FirstOrDefault();

                if (firstFramePath is null)
                    throw new NullReferenceException($"{firstFramePath} is null");

                var targetFaces = await AnalyzeFacesAsync(firstFramePath);

                // todo: this is a silent failure in the case there's no face to swap in the first frame
                if (targetFaces.Any())
                {
                    byte[] result = GetFaceDataOverlayImage(firstFramePath, targetFaces);
                }
            }

            var sourceFaces = msg.FaceFiles.Select(x => x.Face).ToList();

            await Parallel.ForEachAsync(originalFrames, new ParallelOptions { MaxDegreeOfParallelism = 6 }, async (frame, cancellationToken) =>
            {
                var targetFaces = await AnalyzeFacesAsync(frame, AnalysisPackage.SwapOnly);
                var outputImagePath = Path.Combine(msg.ServerSwappedFramesDir, Path.GetFileName(frame));

                if (targetFaces.Any())
                {
                    var targetImage = Cv2.ImRead(frame);
                    Mat img = await SwapFacesAsync(sourceFaces, targetFaces, targetImage);
                    var imageResult = BitmapConverter.ToBitmap(img, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
                    imageResult.Save(outputImagePath);

                    if (msg.RequestImgPreviewEveryNFrames is not null)
                    {
                        byte[] result = File.ReadAllBytes(outputImagePath);
                        var callbackMessage = new JobCallbackMessage(msg, JobResultType.FaceSwapInProgressFrame, result);
                        await msg.TryCallbackAsync(callbackMessage);
                    }
                }
                else
                    File.Copy(frame, outputImagePath);
            });

            if (msg.JobTypes.Count == 1 && msg.JobTypes.First() == JobType.SingleFramePreview)
            {
                var framePreviewPath = Directory.EnumerateFiles(msg.ServerSwappedFramesDir).First();
                var bytes = File.ReadAllBytes(framePreviewPath);
                var callback = new JobCallbackMessage(msg, JobResultType.FacePreviewFrame, bytes, null);
                return callback;
            }

            var outputVideoPath = _fmpegClient.ImageSeriesToVideo(msg.ServerSwappedFramesDir, msg.ServerOutputVideoFilepath, msg.OutputVideoFps);
            var finalCallbackMessage = new JobCallbackMessage(msg, JobResultType.FaceSwappedVideo, File.ReadAllBytes(outputVideoPath), resultFilename: Path.GetFileName (msg.ServerOutputVideoFilepath));

            stopwatch.PrintTime(@"Swap job finished");

            return finalCallbackMessage;

        }

    }

    public string ApplyFirstFrameFaceToAllFramesInVideo(string videoPath)
    {
        var videoFileInfo = _fmpegClient.GetVideoFileInfo(videoPath);
        var tempDir = Directory.CreateTempSubdirectory();
        _fmpegClient.ExportVideoToFrames(videoPath, tempDir.FullName, firstFrameOnly: true, overrideFilename: "first-frame-face.png");

        FileInfo firstFrame = null;
        do
        {
            firstFrame = tempDir.GetFiles().FirstOrDefault();
            Thread.Sleep(10);
        }
        while (firstFrame is null);

        var dir = Path.GetDirectoryName(videoPath);
        return SwapFacesInVideo(firstFrame.FullName, videoPath, dir);
    }

    public string SwapFacesInVideo(string sourceFacePath, string targetVideoPath, string outputVideoDir)
    {
        if (sourceFacePath is null)
            return null;

        var videoFileInfo = _fmpegClient.GetVideoFileInfo(targetVideoPath);

        var tempDir = Directory.CreateTempSubdirectory();
        _fmpegClient.ExportVideoToFrames(targetVideoPath, tempDir.FullName);
        var outputDir = SwapFacesInDir(sourceFacePath, tempDir.FullName, parallelism: 1, prependFaceName: false);
        var videoOutputPath = _fmpegClient.ImageSeriesToVideo(outputDir, "videoOutput.mp4", videoFileInfo.Fps);
        var outputFilename = Path.GetFileNameWithoutExtension(sourceFacePath) + "_" + Path.GetFileNameWithoutExtension(targetVideoPath) + ".mp4";
        var outputFilePath = Path.Combine(outputVideoDir, outputFilename);

        while (!File.Exists(videoOutputPath))
            Thread.Sleep(100);

        File.Copy(videoOutputPath, outputFilePath, true);

        Thread.Sleep(100);
        Directory.Delete(tempDir.FullName, true);

        return outputFilePath;
    }

    public byte[] GetFaceDataOverlayImage(string imagePath, IEnumerable<Face> faces)
    {
        using (var originalImage = new Bitmap(imagePath))
        {
            using (var graphics = Graphics.FromImage(originalImage))
            {
                foreach (var face in faces)
                {
                    // Draw bounding box
                    using (var rectanglePen = new Pen(Color.White))
                    {
                        graphics.DrawRectangle(rectanglePen, face.BoundingBox.Rectangle);
                    }

                    // Draw key points
                    var kps = face.KeyFaceFeatures;
                    DrawKeyPoint(graphics, kps.LeftEye, Color.Blue);
                    DrawKeyPoint(graphics, kps.RightEye, Color.Red);
                    DrawKeyPoint(graphics, kps.Nose, Color.Black);
                    DrawKeyPoint(graphics, kps.LeftMouthCorner, Color.Violet);
                    DrawKeyPoint(graphics, kps.RightMouthCorner, Color.Orange);

                    // Add overlay text
                    using (var overlayFont = new Font("Arial", 9, FontStyle.Regular))
                    using (var overlayBrush = new SolidBrush(Color.White))
                    {
                        var textPosition = new PointF(face.BoundingBox.Rectangle.Left + 8, face.BoundingBox.Rectangle.Bottom - 58);
                        var overlay = $"S | {face.FaceSizePercentage.ToString("P0")}\nY | {face.Pose.Yaw.ToString("F0")}°\nP | {face.Pose.Pitch.ToString("F0")}°\nR | {face.Pose.Roll.ToString("F0")}°";
                        graphics.DrawString(overlay, overlayFont, overlayBrush, textPosition);
                    }
                }
            }

            var bytes = ImageToByteArray(originalImage);

            return bytes;
        }
    }

    private void DrawKeyPoint(Graphics graphics, PointF point, Color color)
    {
        using (var pointPen = new Pen(color))
        {
            graphics.DrawEllipse(pointPen, point.X - 2, point.Y - 2, 4, 4); // Centered ellipse
        }
    }

    static byte[] ImageToByteArray(Bitmap bitmap, ImageFormat format = null)
    {
        format ??= ImageFormat.Jpeg;

        using (var memoryStream = new MemoryStream())
        {
            bitmap.Save(memoryStream, format);
            return memoryStream.ToArray();
        }
    }

    List<string> GetSourceFaceImgPaths(FaceSwapJob msg) => msg.FaceFiles.Select(x => Path.Combine(_options.LocalImageDir, x.IsGlobal ? "global" : msg.Username, x.FileName)).ToList();

    OnnxModel[] GetAnalysisPackage(AnalysisPackage package)
    {
        return package switch
        {
            AnalysisPackage.Full => [OnnxModel.RetinaFace, OnnxModel.Landmark2d, OnnxModel.Landmark3d, OnnxModel.GenderAge, OnnxModel.Arcface, OnnxModel.InSwapper],
            AnalysisPackage.Essential => [OnnxModel.RetinaFace, OnnxModel.Landmark3d, OnnxModel.GenderAge, OnnxModel.Arcface, OnnxModel.InSwapper],
            AnalysisPackage.SwapOnly => [OnnxModel.RetinaFace, OnnxModel.Arcface, OnnxModel.InSwapper],
            _ => [],
        };
    }
}
