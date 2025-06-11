using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Net.WebSockets;
using System.Runtime.InteropServices;

namespace FaceSwapOnnx;

public static class Extensions
{
    public static unsafe DenseTensor<float> ToDenseTensor(this Mat blob)
    {
        if (blob.Type() != MatType.CV_32F)
            throw new NotSupportedException($"Expected blob of type CV_32F, but got {blob.Type()}.");

        // Get dimensions
        int batchSize = blob.Size(0);
        int channels = blob.Size(1);
        int height = blob.Size(2);
        int width = blob.Size(3);

        // Calculate total number of elements
        int totalElements = batchSize * channels * height * width;

        // Get pointer to the blob data
        float* floatPtr = (float*)blob.Data.ToPointer();

        // Create a Span<float> over the unmanaged memory
        Span<float> floatSpan = new Span<float>(floatPtr, totalElements);

        fixed (float* ptr = floatSpan)
        {
            var memoryManager = new UnmanagedMemoryManager<float>(ptr, floatSpan.Length);
            Memory<float> memory = memoryManager.Memory;

            // Create the DenseTensor
            var tensor = new DenseTensor<float>(memory, new[] { batchSize, channels, height, width });

            return tensor;
        }
    }

    public static DenseTensor<float> GetDummyDenseTensor(int size)
    {
        int batchSize = 1;
        int channels = 3;
        int height = size;
        int width = size;

        // Create dummy data filled with zeros or any other value
        float[] dummyData = new float[batchSize * channels * height * width];

        return new DenseTensor<float>(dummyData, new[] { batchSize, channels, height, width });
    }

    public static DenseTensor<float> GetDummyData(params int[] shape)
    {
        int[] adjustedShape = shape.Select(Math.Abs).ToArray();
        int totalElements = adjustedShape.Aggregate(1, (a, b) => a * b);
        float[] dummyData = new float[totalElements];
        return new DenseTensor<float>(dummyData, adjustedShape);
    }

    public static DenseTensor<float> GetDummyDataForFirstInput()
    {
        // Define the shape based on the buffer size and rank
        int[] shape = { 1, 3, 128, 128 };
        int totalElements = shape.Aggregate(1, (a, b) => a * b);

        // Create dummy data (e.g., zeros or any other value)
        float[] dummyData = new float[totalElements];

        // Create and return the DenseTensor
        return new DenseTensor<float>(dummyData, shape);
    }

    public static DenseTensor<float> GetDummyDataForSecondInput()
    {
        // Define the shape based on the buffer size and rank
        int[] shape = { 1, 512 };
        int totalElements = shape.Aggregate(1, (a, b) => a * b);

        // Create dummy data (e.g., zeros or any other value)
        float[] dummyData = new float[totalElements];

        // Create and return the DenseTensor
        return new DenseTensor<float>(dummyData, shape);
    }



    public static double[] ToDouble(this int[] array)
    {
        var newArray = new double[array.Length];

        for (int i = 0; i < array.Length; i++)
            newArray[i] = array[i];

        return newArray;
    }

    public static double[] ToDouble(this float[] array)
    {
        var newArray = new double[array.Length];

        for (int i = 0; i < array.Length; i++)
            newArray[i] = array[i];

        return newArray;
    }

    public static Mat ExtractValuesByIndices(this Mat input, List<int> indices)
    {
        Mat result = new Mat(indices.Count, 1, input.Type());

        for (int i = 0; i < indices.Count; i++)
            result.Set<float>(i, input.At<float>(indices[i]));

        return result;
    }

    public static Mat VStack(this List<Mat> mats)
    {
        Mat result = new();
        Cv2.VConcat(mats, result);
        return result;
    }

    public static Mat HStack(this List<Mat> mats)
    {
        Mat result = new();
        Cv2.HConcat(mats, result);
        return result;
    }

    public static Mat VStack(params Mat[] mats)
    {
        Mat result = new();
        Cv2.VConcat(mats, result);
        return result;
    }

    public static Mat HStack(params Mat[] mats)
    {
        Mat result = new();
        Cv2.HConcat(mats, result);
        return result;
    }

    public static Mat ConvertChannelsToColumns(this Mat input)
    {
        if (input.Cols != 1)
        {
            throw new ArgumentException("Input Mat must have 1 column.");
        }

        // Reshape the Mat: the second parameter is the new row count (-1 keeps the row count the same)
        Mat reshaped = input.Reshape(1, input.Rows); // 1 channel, row count remains 1
        return reshaped;
    }

    public static Mat AppendAdditionalChannels(this Mat mat1, Mat mat2)
    {
        if (mat1.Size() != mat2.Size())
            throw new ArgumentException("Both Mats must have the same size (rows and columns).");

        if (mat1.Depth() != mat2.Depth())
            throw new ArgumentException("Both Mats must have the same depth (data type).");

        // Split channels of both Mats
        Mat[] mat1Channels = mat1.Split();
        Mat[] mat2Channels = mat2.Split();

        // Combine channels
        Mat[] combinedChannels = new Mat[mat1Channels.Length + mat2Channels.Length];
        mat1Channels.CopyTo(combinedChannels, 0);
        mat2Channels.CopyTo(combinedChannels, mat1Channels.Length);

        // Merge into a new Mat
        Mat output = new Mat();
        Cv2.Merge(combinedChannels, output);

        return output;
    }

    public static Mat ToMultiChannel(this Mat input, int numChannels)
    {
        if (input.Channels() != 1)
            throw new ArgumentException("Input Mat must be a single-channel matrix.", nameof(input));

        if (numChannels < 1)
            throw new ArgumentException("Number of channels must be at least 1.", nameof(numChannels));

        // Create an array of Mats to hold repeated channels
        Mat[] channels = new Mat[numChannels];
        for (int i = 0; i < numChannels; i++)
            channels[i] = input;

        // Merge the single channel into the desired number of channels
        Mat output = new Mat();
        Cv2.Merge(channels, output);

        return output;
    }

    public static Mat ReorderMat(this Mat inputMat, int[] indexes)
    {
        // Validate input
        if (indexes.Length != inputMat.Rows)
            throw new ArgumentException("Index array length must match the number of rows in the Mat.");

        Mat outputMat = new Mat(inputMat.Rows, inputMat.Cols, inputMat.Type());

        for (int i = 0; i < indexes.Length; i++)
        {
            if (indexes[i] < 0 || indexes[i] >= inputMat.Rows)
                throw new ArgumentOutOfRangeException($"Index {indexes[i]} is out of range.");

            // Copy the specified row to the output Mat
            inputMat.Row(indexes[i]).CopyTo(outputMat.Row(i));
        }

        return outputMat;
    }


    /*    public static Mat ToChanneledMat(this DisposableNamedOnnxValue namedValue, int widthPerChannel = 1)
        {
            var tensor = namedValue.AsTensor<float>();
            float[] data = tensor.ToArray();

            var dimensions = tensor.Dimensions.ToArray();

            // Ensure the tensor exactly two dimensions
            if (dimensions.Length != 2)
                throw new InvalidOperationException($"Expected tensor to have 2 dimensions, but found {dimensions.Count()}");

            int channels = dimensions[1];
            int height = dimensions[0];
            int width = widthPerChannel;

            Mat mat = new Mat(height, 1, MatType.CV_32FC(channels));

            // Step 4: Copy data into the Mat
            Marshal.Copy(data, 0, mat.Data, data.Length);

            return mat;
        }*/

    /*    public static Mat ToChanneledMat(this DisposableNamedOnnxValue namedValue, bool swapAxes = false, int? segmentIntoChannelSize = null)
        {
            var tensor = namedValue.AsTensor<float>();
            float[] data = tensor.ToArray();

            var dimensions = tensor.Dimensions.ToArray();

            // Ensure the tensor exactly two dimensions
            if (dimensions.Length != 2)
                throw new InvalidOperationException($"Expected tensor to have 2 dimensions, but found {dimensions.Count()}");

            int channelDimension = swapAxes ? 0 : 1;
            int heightDimension = swapAxes ? 1 : 0;

            int channels = dimensions[channelDimension];
            int height = dimensions[heightDimension];

            Mat resultMat = new();

            if (segmentIntoChannelSize is not null)
            {
                List<Mat> channelMats = new();
                channels = segmentIntoChannelSize.Value;
                height /= channels;
                var segmentedData = data.SegmentIntoChannelSize(channels);

                for (int i = 0; i < channels; i++)
                {
                    Mat channelMat = new(height, 1, MatType.CV_32FC1);
                    Marshal.Copy(segmentedData[i], 0, channelMat.Data, height);
                    channelMats.Add(channelMat);
                }

                Cv2.Merge(channelMats.ToArray(), resultMat);
            }
            else
            {
                resultMat = new Mat(height, 1, MatType.CV_32FC(channels));
                Marshal.Copy(data, 0, resultMat.Data, data.Length);
            }

            return resultMat;
        }*/

    public static Mat ToChanneledMat(this DisposableNamedOnnxValue namedValue, bool swapAxes = false, int? segmentIntoChannelSize = null)
    {
        var tensor = namedValue.AsTensor<float>();
        float[] data = tensor.ToArray();

        var dimensions = tensor.Dimensions.ToArray();

        // Ensure the tensor exactly two dimensions
        if (dimensions.Length != 2)
            throw new InvalidOperationException($"Expected tensor to have 2 dimensions, but found {dimensions.Count()}");

        int channelDimension = swapAxes ? 0 : 1;
        int heightDimension = swapAxes ? 1 : 0;

        int height = dimensions[heightDimension];

        Mat resultMat = new();

        if (segmentIntoChannelSize is not null)
        {
            /*            List<Mat> channelMats = new();
                        channels = segmentIntoChannelSize.Value;
                        height /= channels;

                        var channelSegmentedRows = data.SegmentIntoChannelSize(channels);

                        for (int row = 0; row < height; row++)
                        {
                            for (int j = 0; j < channels; j++)
                            {
                                var rowData = channelSegmentedRows[row];
                            }
                        }

                        for (int i = 0; i < channels; i++)
                        {
                            Mat channelMat = new(height, 1, MatType.CV_32FC1);
                            Marshal.Copy(channelSegmentedRows[i, 0..1], 0, channelMat.Data, height);
                            channelMats.Add(channelMat);
                        }

                        for (int i = 0; i < channels; i++)
                        {
                            Mat channelMat = new(height, 1, MatType.CV_32FC1);
                            Marshal.Copy(channelSegmentedRows[i], 0, channelMat.Data, height);
                            channelMats.Add(channelMat);
                        }

                        Cv2.Merge(channelMats.ToArray(), resultMat);*/

            // Calculate the number of rows
            int rows = data.Length / segmentIntoChannelSize.Value;

            // Create a Mat with the specified rows and channels
            resultMat = new Mat(rows, 1, MatType.CV_32FC(segmentIntoChannelSize.Value));

            // Copy the data directly into the Mat's buffer
            unsafe
            {
                // Get the pointer to the Mat's data
                float* matData = (float*)resultMat.Data.ToPointer();

                // Use a single copy operation for performance
                for (int i = 0; i < data.Length; i++)
                {
                    matData[i] = data[i];
                }
            }
        }
        else
        {
            int channels = dimensions[channelDimension];
            resultMat = new Mat(height, 1, MatType.CV_32FC(channels));
            Marshal.Copy(data, 0, resultMat.Data, data.Length);
        }

        return resultMat;
    }



    public static Mat ToDimensionalMat(this DisposableNamedOnnxValue namedValue)
    {
        // Step 1: Extract tensor data as a flat float array
        var tensor = namedValue.AsTensor<float>();
        float[] data = tensor.ToArray();

        // Step 2: Retrieve dimensions of the tensor
        var dimensions = tensor.Dimensions.ToArray();

        // Ensure the tensor has at least two dimensions (e.g., H x W or C x H x W)
        if (dimensions.Length < 2)
            throw new InvalidOperationException("Tensor must have at least 2 dimensions to create a Mat.");

        // Handle the tensor shape. For example: (C, H, W) => (H, W, C) for OpenCV.
        int channels = dimensions.Length == 3 ? dimensions[0] : 1; // If shape is 3D, first dim = channels.
        int height = dimensions[dimensions.Length - 2];
        int width = dimensions[dimensions.Length - 1];

        // Step 3: Create a Mat object
        MatType matType = channels == 1 ? MatType.CV_32F : MatType.CV_32FC(channels);
        Mat mat = new Mat(height, width, matType);

        // Step 4: Copy data into the Mat
        Marshal.Copy(data, 0, mat.Data, data.Length);

        return mat;
    }

    /*    public static List<float[]> SegmentIntoChannelSize(this float[] data, int size)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (size <= 0)
                throw new ArgumentException("Size must be greater than zero.", nameof(size));
            if (data.Length % size != 0)
                throw new ArgumentException("The length of the data must be divisible by the size.", nameof(data));

            int chunkLength = data.Length / size;
            List<float[]> result = new List<float[]>(size);

            for (int i = 0; i < size; i++)
            {
                float[] chunk = new float[chunkLength];
                Array.Copy(data, i * chunkLength, chunk, 0, chunkLength);
                result.Add(chunk);
            }

            return result;
        }*/

    public static Mat DotProduct(this Mat mat1, Mat mat2)
    {
        // Perform matrix multiplication using Cv2.Gemm
        Mat result = new Mat();
        Mat emptyMat = Mat.Zeros(mat1.Rows, mat2.Cols, mat1.Type()); // Create an empty Mat for src3
        Cv2.Gemm(mat1.T(), mat2, 1.0, emptyMat, 0.0, result);

        return result;
    }

    public static Mat MultiplyJagged(this Mat multiColumnMat, Mat singleColumnMat)
    {
        // Validate input dimensions
        if (singleColumnMat.Cols != 1)
            throw new ArgumentException("The second matrix must have exactly one column.");
        if (singleColumnMat.Rows != multiColumnMat.Rows)
            throw new ArgumentException("Both matrices must have the same number of rows.");

        // Repeat the columnMat across the number of columns in multiColumnMat
        Mat expandedColumnMat = new Mat();
        Cv2.Repeat(singleColumnMat, 1, multiColumnMat.Cols, expandedColumnMat);

        // Perform element-wise multiplication
        Mat result = new Mat();
        Cv2.Multiply(multiColumnMat, expandedColumnMat, result);

        return result;
    }

    public static Mat DuplicateHorizontal(this Mat inputMat, int duplicationCount)
    {
        if (duplicationCount <= 0)
        {
            throw new ArgumentException("Duplication count must be greater than zero.", nameof(duplicationCount));
        }

        // Prepare a list of Mats for horizontal stacking
        Mat[] duplicatedMats = new Mat[duplicationCount];
        for (int i = 0; i < duplicationCount; i++)
        {
            duplicatedMats[i] = inputMat.Clone(); // Clone the input Mat to avoid modifying the original
        }

        // Horizontally stack the duplicated Mats
        Mat result = new Mat();
        Cv2.HConcat(duplicatedMats, result);

        return result;
    }

    public static double L2Norm(this Mat mat)
    {
        // Compute the sum of squared elements
        double sumOfSquares = Cv2.Sum(mat.Mul(mat)).Val0;

        // Return the square root of the sum of squares
        return Math.Sqrt(sumOfSquares);
    }

    public static float[,] SegmentIntoChannelSize(this float[] data, int size)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        if (size <= 0)
            throw new ArgumentException("Size must be greater than zero.", nameof(size));
        if (data.Length % size != 0)
            throw new ArgumentException("The length of the data must be divisible by the size.", nameof(data));

        int rows = data.Length / size;
        float[,] result = new float[rows, size];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < size; j++)
                result[i, j] = data[i * size + j];

        return result;
    }

    //public static double[] GetBlobFromImage(Bitmap img, double scalefactor, (int, int) size, double mean, bool swapRB)
    //{
    //    // Equivalent to cv2.dnn.blobFromImage
    //    int width = size.Item1;
    //    int height = size.Item2;
    //    Bitmap resizedImg = new Bitmap(img, new System.Drawing.Size(width, height));
    //    double[] blob = new double[1 * 3 * height * width];
    //
    //    for (int y = 0; y < height; y++)
    //    {
    //        for (int x = 0; x < width; x++)
    //        {
    //            Color pixel = resizedImg.GetPixel(x, y);
    //
    //            // Normalize pixel values
    //            double b = (pixel.B - mean) * scalefactor;
    //            double g = (pixel.G - mean) * scalefactor;
    //            double r = (pixel.R - mean) * scalefactor;
    //
    //            int idx = y * width + x;
    //            blob[idx] = swapRB ? r : b; // Channel 0
    //            blob[height * width + idx] = g; // Channel 1
    //            blob[2 * height * width + idx] = swapRB ? b : r; // Channel 2
    //        }
    //    }
    //
    //    return blob;
    //}

    public static double[] MatsToFlat(this List<Mat> mats) => mats.SelectMany(MatToFlat).ToArray();

    //public static double[] BlobMat2DToFlat(this Mat blobMat)
    //{
    //    if (blobMat.Dims != 2)
    //        throw new ArgumentException("Expected a 2-dimensional Mat.");
    //
    //    if (blobMat.Total() == 0)
    //        return new double[] { };
    //
    //    // Get the dimensions of the blobMat
    //    int b = blobMat.Size(0); // Batches (or "slices")
    //    int v = blobMat.Size(1); // Values per slice
    //
    //    // Allocate a flat array to hold all elements
    //    double[] flatArray = new double[b * v];
    //
    //    // Copy data directly from Mat to flatArray
    //    float[] blobData = new float[blobMat.Total()];
    //    Marshal.Copy(blobMat.Data, blobData, 0, blobData.Length);
    //
    //    // Convert float[] to double[] for the output
    //    for (int i = 0; i < blobData.Length; i++)
    //    {
    //        flatArray[i] = blobData[i];
    //    }
    //
    //    return flatArray;
    //}

    public static double[] MatToFlat(this Mat mat)
    {
        var type = mat.Type();
        int dims = mat.Dims;
        int width = mat.Width;
        int rows = mat.Rows;
        int cols = mat.Cols;
        int channels = mat.Channels();

        if (type == MatType.CV_8UC3)
        {
            // img mat from bitmap

            mat.GetRectangularArray(out Vec3b[,] array);

            int rectRows = array.GetLength(0);
            int rectCols = array.GetLength(1);

            Vec3b[] flatBytes = new Vec3b[rectRows * rectCols];
            int index = 0;

            for (int row = 0; row < rectRows; row++)
                for (int col = 0; col < rectCols; col++)
                    flatBytes[index++] = array[row, col];

            int[] flatInts = new int[flatBytes.Length * 3];
            for (int i = 0; i < flatBytes.Length; i++)
            {
                int flatIndex = i * 3;
                flatInts[flatIndex + 0] = flatBytes[i].Item0;
                flatInts[flatIndex + 1] = flatBytes[i].Item1;
                flatInts[flatIndex + 2] = flatBytes[i].Item2;
            }

            return flatInts.ToDouble();
        }

        else if (type == MatType.CV_32FC(channels))
        {
            if (dims == 4 && channels == 1)
            {
                // Cv2 blob from img

                int n = mat.Size(0); // Batch size (usually 1)
                int c = mat.Size(1); // Channels (referring to "channels" in the NCHW format sense, different from above)
                int h = mat.Size(2); // Height
                int w = mat.Size(3); // Width

                // Allocate a flat array to hold all elements
                double[] flatArray = new double[n * c * h * w];

                // Copy data directly from Mat to flatArray
                float[] blobData = new float[mat.Total()];
                Marshal.Copy(mat.Data, blobData, 0, blobData.Length);

                return blobData.ToDouble();
            }
            else if (dims == 2)
            {
                // from calling.ToChanneledMat to produce a single-column array of floats

                double[] flatArray = new double[rows * cols * channels];
                Mat[] channelMats = Cv2.Split(mat);

                // interleave channel data
                int i = 0;
                for (int row = 0; row < rows; row++)
                    for (int col = 0; col < cols; col++)
                        for (int channel = 0; channel < channels; channel++)
                        {
                            flatArray[i++] = channelMats[channel].At<float>(row, col);
                        }

                return flatArray;
            }
        }

        return [];
    }

    public static void OverlayAtTopLeftCorner(this Mat underlyingImg, Mat overlayingImg, int? height = null, int? width = null)
    {
        if (height is null || width is null)
        {
            height = Math.Min(underlyingImg.Height, overlayingImg.Height);
            width = Math.Min(underlyingImg.Width, overlayingImg.Width);
        }

        // sub-area of underlying image image
        Rect subArea = new Rect(0, 0, width.Value, height.Value);
        Mat subAreaContentOfUnderlying = new Mat(underlyingImg, subArea);
        overlayingImg.CopyTo(subAreaContentOfUnderlying);
    }

    public static double Norm(double[] x)
    {
        // Compute the square of each element, sum them, and take the square root
        return (double)Math.Sqrt(x.Select(val => val * val).Sum());
    }

    public static Mat ScaleAndReshape(this Mat meshGrid, int stride)
    {
        if (meshGrid.Channels() != 2)
            throw new ArgumentException("Expected a Mat with 2 channels (x, y coordinates).");

        // Multiply each element by the stride
        Mat scaledGrid = new Mat();
        Cv2.Multiply(meshGrid, new Scalar(stride, stride), scaledGrid); // Element-wise multiplication

        // Reshape to (-1, 2)
        int totalPoints = meshGrid.Rows * meshGrid.Cols; // Total number of points
        Mat reshapedGrid = scaledGrid.Reshape(2, totalPoints); // Reshape to (6400, 2) for 80x80 grid

        return reshapedGrid;
    }

    public static Mat ExpandAndReshape(this Mat anchorCenters, int numAnchors)
    {
        if (anchorCenters.Channels() != 2)
            throw new ArgumentException("Expected a Mat with 2 channels (x, y coordinates).");

        // Step 1: Expand by repeating anchorCenters for numAnchors
        List<Mat> repeatedAnchors = new List<Mat>();
        for (int i = 0; i < numAnchors; i++)
        {
            repeatedAnchors.Add(anchorCenters.Clone());
        }

        // Merge the repeated Mats along a new axis (channel axis)
        Mat expandedAnchors = new Mat();
        Cv2.Merge(repeatedAnchors.ToArray(), expandedAnchors);

        // Step 2: Reshape to (-1, 2)
        int totalRows = anchorCenters.Rows * numAnchors;
        Mat reshapedAnchors = expandedAnchors.Reshape(2, totalRows);

        return reshapedAnchors;
    }

    public static OnnxModel[] GetAnalysisPackage(AnalysisPackage package)
    {
        return package switch
        {
            AnalysisPackage.Full => [OnnxModel.RetinaFace, OnnxModel.Landmark2d, OnnxModel.Landmark3d, OnnxModel.GenderAge, OnnxModel.Arcface, OnnxModel.InSwapper],
            AnalysisPackage.Essential => [OnnxModel.RetinaFace, OnnxModel.Landmark3d, OnnxModel.GenderAge, OnnxModel.Arcface, OnnxModel.InSwapper],
            AnalysisPackage.SwapOnly => [ OnnxModel.RetinaFace, OnnxModel.Landmark3d, OnnxModel.Arcface, OnnxModel.InSwapper ],
            _ => new OnnxModel[0],
        };
    }

    public static void PrintTime(this Stopwatch stopwatch, string message) => Console.WriteLine($"{message} in {stopwatch.Elapsed.TotalSeconds:0.##}s");
}
