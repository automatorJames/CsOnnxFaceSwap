namespace FaceSwapOnnx.Ffmpeg;
public class VideoFileInfo
{
    public string FilePath { get; set; }
    public string Filename { get; set; }
    public TimeSpan Duration { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public int Kbps { get; set; }
    public double Fps { get; set; }

    public VideoFileInfo(string ffmpegInfoString)
    {
        FilePath = Regex.Match(ffmpegInfoString, @"(?<=from ').+(?=':)").Value;
        Filename = Regex.Match(FilePath, @"([^\\]+$)").Value;
        Duration = TimeSpan.Parse(Regex.Match(ffmpegInfoString, @"(?<=Duration: ).+?(?=,)").Value);
        Width = int.Parse(Regex.Match(ffmpegInfoString, @"(?<=Stream.+[ ])\d{2,5}(?=x\d{2,5}([ ]|,))").Value);
        Height = int.Parse(Regex.Match(ffmpegInfoString, @"(?<=Stream.+[ ]\d{2,5}x)\d{2,5}(?=([ ]|,))").Value);
        Kbps = int.Parse(Regex.Match(ffmpegInfoString, @"\d{2,6}(?= kb\/s)").Value);

        if (Regex.IsMatch(ffmpegInfoString, @"(\d|\.){1,6}(?= fps)"))
            Fps = double.Parse(Regex.Match(ffmpegInfoString, @"(\d|\.){1,6}(?= fps)").Value);
        else
            Fps = double.Parse(Regex.Match(ffmpegInfoString, @"(\d|\.){1,6}(?= tbr)").Value);
    }
}