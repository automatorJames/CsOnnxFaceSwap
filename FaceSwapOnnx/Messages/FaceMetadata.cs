namespace FaceSwapOnnx.Messages;

public class FaceMetadata
{
    public string FaceName { get; set; }
    public string ClientSideFilePath { get; set; }
    public string ServerSideFilePath { get; set; }
    public string FileName { get; set; }
    public Face Face { get; set; }
    public bool IsGlobal { get; set; }
    public bool IsInvalid { get; set; }
    public bool IsNewlyUploaded { get; set; }
    public byte[] ImgData { get; set; }

    public FaceMetadata()
    {
    }

    public FaceMetadata(string filePath, bool isGlobal = false, bool readImgData = false)
    {
        ClientSideFilePath = filePath;
        FileName = Path.GetFileName(filePath);
        FaceName = Path.GetFileNameWithoutExtension(FileName);
        IsGlobal = isGlobal;

        if (readImgData)
            SetImgData();
    }

    public void SetImgData()
    {
        if (ClientSideFilePath is null)
            throw new Exception($"Can't set ImgData because {nameof(ClientSideFilePath)} is null");

        ImgData = File.ReadAllBytes(ClientSideFilePath);
    }
}
