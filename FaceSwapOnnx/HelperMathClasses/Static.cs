namespace FaceSwapOnnx.HelperMathClasses;

public static class Static
{
    public static Mat GetBinaryMatrix(BinaryMatrix matrixName)
    {
        var binaryFilePath = Path.Combine(AppContext.BaseDirectory, "BinaryMatrices", BinaryFilenames[matrixName]);

        double[,] data = LoadBinaryFile(binaryFilePath);

        Mat mat = new Mat(data.GetLength(0), data.GetLength(1), MatType.CV_32F);
        for (int i = 0; i < data.GetLength(0); i++)
            for (int j = 0; j < data.GetLength(1); j++)
                mat.Set(i, j, (float)data[i, j]);

        return mat;
    }

    static Dictionary<BinaryMatrix, string> BinaryFilenames = new Dictionary<BinaryMatrix, string>
    {
        { BinaryMatrix.EMap, "emap.bin" },
        { BinaryMatrix.MeanShape, "meanshape.bin" },
    };

    static double[,] LoadBinaryFile(string filePath)
    {
        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (var br = new BinaryReader(fs))
        {
            // Read dimensions
            int rows = br.ReadInt32();
            int cols = br.ReadInt32();

            // Initialize the 2D array
            var emap = new double[rows, cols];

            // Populate the array
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    emap[i, j] = br.ReadDouble();

            return emap;
        }
    }
}
