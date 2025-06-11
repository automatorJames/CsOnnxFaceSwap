using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;

namespace FaceSwapOnnx;

public static class FaceAlign
{
    public static void Transform(Mat img, Tuple<double, double> center, int outputSize, double scale, double rotate, out Mat transformedImg, out Mat M)
    {
        double scaleRatio = scale;
        double rot = rotate * Math.PI / 180.0;

        // Step 1: Scaling Transformation (t1)
        Mat t1 = GetScalingMatrix(scaleRatio);

        // Step 2: Translation to Center (t2)
        double cx = center.Item1 * scaleRatio;
        double cy = center.Item2 * scaleRatio;
        Mat t2 = GetTranslationMatrix(-cx, -cy);

        // Step 3: Rotation Transformation (t3)
        Mat t3 = GetRotationMatrix(rot);

        // Step 4: Translation to Output Center (t4)
        Mat t4 = GetTranslationMatrix(outputSize / 2.0, outputSize / 2.0);

        // Combine transformations: t = t4 * t3 * t2 * t1
        Mat t = t4 * t3 * t2 * t1;

        // Extract the affine transformation matrix M (first two rows)
        M = new Mat(2, 3, MatType.CV_64F);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                M.Set<double>(i, j, t.At<double>(i, j));
            }
        }

        // todo: just fix GetAffineTransform to return floats?
        M.ConvertTo(M, MatType.CV_32FC1);

        // Apply affine transformation to the image
        transformedImg = new Mat();
        Cv2.WarpAffine(
            src: img,
            dst: transformedImg,
            m: M,
            dsize: new Size(outputSize, outputSize),
            borderValue: Scalar.All(0)
        );
    }

    private static Mat GetScalingMatrix(double scale)
    {
        Mat t = Mat.Eye(3, 3, MatType.CV_64F);
        t.Set<double>(0, 0, scale);
        t.Set<double>(1, 1, scale);
        return t;
    }

    private static Mat GetTranslationMatrix(double tx, double ty)
    {
        Mat t = Mat.Eye(3, 3, MatType.CV_64F);
        t.Set<double>(0, 2, tx);
        t.Set<double>(1, 2, ty);
        return t;
    }

    private static Mat GetRotationMatrix(double angleRad)
    {
        double cosA = Math.Cos(angleRad);
        double sinA = Math.Sin(angleRad);

        Mat r = Mat.Eye(3, 3, MatType.CV_64F);
        r.Set<double>(0, 0, cosA);
        r.Set<double>(0, 1, -sinA);
        r.Set<double>(1, 0, sinA);
        r.Set<double>(1, 1, cosA);

        return r;
    }

    private static Mat GetAffineTransform(Tuple<double, double> center, double scale, double rotate, int outputSize)
    {
        // Compute the affine transformation matrix M
        double angle = rotate * Math.PI / 180.0;
        double cos = Math.Cos(angle);
        double sin = Math.Sin(angle);

        double scaleInv = 1.0 / scale;

        double tx = -center.Item1 * cos - center.Item2 * sin + (outputSize / 2.0);
        double ty = center.Item1 * sin - center.Item2 * cos + (outputSize / 2.0);

        // Create a 2x3 affine transformation matrix
        Mat M = new Mat(2, 3, MatType.CV_32F);
        M.Set<double>(0, 0, cos * scaleInv);
        M.Set<double>(0, 1, sin * scaleInv);
        M.Set<double>(0, 2, tx);
        M.Set<double>(1, 0, -sin * scaleInv);
        M.Set<double>(1, 1, cos * scaleInv);
        M.Set<double>(1, 2, ty);

        return M;
    }

    public static Mat InvertAffineTransform(Mat M)
    {
        if (M.Rows != 2 || M.Cols != 3)
            throw new ArgumentException("Input matrix M must be of size 2x3.");

        Mat M_inv = new Mat();
        Cv2.InvertAffineTransform(M, M_inv);

        return M_inv;
    }

    public static Mat TransPoints(Mat points, Mat M)
    {
        int numPoints = points.Rows;
        int pointDim = points.Cols;

        // Use only the first two columns of points
        Mat points2D = points.ColRange(0, 2).Clone(); // (n x 2)

        // Build homogeneous coordinates for 2D points
        Mat ones = Mat.Ones(numPoints, 1, type: MatType.CV_64F);
        Mat pointsHomo = new Mat();
        Cv2.HConcat(new Mat[] { points2D, ones }, pointsHomo); // (n x 3)

        // Multiply by M transpose
        Mat M_T = M.T(); // Transpose of M (3 x 2)
        Mat transformedPoints2D = pointsHomo * M_T; // (n x 3) x (3 x 2) = (n x 2)

        if (pointDim > 2)
        {
            // If original points had more dimensions, concatenate them back
            Mat remainingColumns = points.ColRange(2, pointDim).Clone(); // (n x (pointDim - 2))
            Mat result = new Mat();
            Cv2.HConcat(new Mat[] { transformedPoints2D, remainingColumns }, result);
            return result;
        }
        else
        {
            return transformedPoints2D;
        }
    }

    // ArcFace default destination landmarks
    private static readonly Point2f[] ArcFaceDst = new Point2f[]
    {
        new Point2f(38.2946f, 51.6963f),
        new Point2f(73.5318f, 51.5014f),
        new Point2f(56.0252f, 71.7366f),
        new Point2f(41.5493f, 92.3655f),
        new Point2f(70.7299f, 92.2041f)
    };

    public static Mat EstimateNorm(Mat lmk, int imageSize = 112, string mode = "arcface")
   {
        // Ensure the landmark shape is valid
        if (lmk.Rows != 5 || lmk.Cols != 2)
            throw new ArgumentException("Landmark shape must be (5, 2)");

        if (imageSize % 112 != 0 && imageSize % 128 != 0)
            throw new ArgumentException("Image size must be a multiple of 112 or 128");

        float ratio = (imageSize % 112 == 0) ? (float)imageSize / 112.0f : (float)imageSize / 128.0f;
        float diffX = (imageSize % 112 == 0) ? 0 : 8.0f * ratio;

        Mat dst = GetDST(ratio, diffX);

        // Create a similarity transform
        Mat M = new Mat(2, 3, MatType.CV_32F); // 2x3 affine transform matrix

        var transform = new SimilarityTransform();
        if (transform.Estimate(lmk, dst, estimateScale: true))
        {
            //Console.WriteLine("Transformation matrix:");
            //Console.WriteLine(transform.Params.Dump());
        }

        // drop the last column from Params
        M = new Mat(transform.Params, new Rect(0, 0, transform.Params.Cols, 2));

        return M;
    }

    public static Mat GetDST(float ratio, float diffX)
    {
        Mat dst = new Mat(5, 2, MatType.CV_32F);

        // Populate the Mat with data
        for (int row = 0; row < dst.Rows; row++)
        {
            for (int col = 0; col < dst.Cols; col++)
            {
                dst.Set(row, col, ArcfaceDstData[row, col]);
            }
        }

        dst *= ratio;

        // Adjust the first column (x-coordinates) by diff_x
        for (int row = 0; row < dst.Rows; row++)
        {
            float x = dst.At<float>(row, 0);
            dst.Set(row, 0, x + diffX);
        }

        return dst;
    }

    private static readonly float[,] ArcfaceDstData =
    {
        { 38.2946f, 51.6963f },
        { 73.5318f, 51.5014f },
        { 56.0252f, 71.7366f },
        { 41.5493f, 92.3655f },
        { 70.7299f, 92.2041f }
    };

    private static void StubTformEstimate(Mat lmk, Mat dst, Mat M)
    {
        // Stub for estimating the transformation matrix
        // You can implement the actual logic here as per SimilarityTransform in Python
        throw new NotImplementedException("Transform estimate logic needs to be implemented.");
    }

    public static Mat NormCrop(Mat img, Mat landmark, int imageSize = 112, string mode = "arcface")
    {
        Mat M = EstimateNorm(landmark, imageSize, mode);

        // Perform affine warp
        Mat warped = new Mat();
        Cv2.WarpAffine(img, warped, M, new Size(imageSize, imageSize));
        return warped;
    }

    public static (Mat, Mat) NormCrop2(Mat img, Mat landmark, int imageSize = 112, string mode = "arcface")
    {
        Mat M = EstimateNorm(landmark, imageSize, mode);

        // Apply the affine transformation
        Mat alignedImg = new Mat();
        Cv2.WarpAffine(img, alignedImg, M, new OpenCvSharp.Size(imageSize, imageSize));

        return (alignedImg, M);
    }

    private static Point2f[] GetReferencePoints(int image_size)
    {
        // Define reference points for alignment (adjust according to your model)
        float[,] reference = new float[,]
        {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}
        };

        Point2f[] dstPoints = new Point2f[5];
        for (int i = 0; i < 5; i++)
        {
            float x = reference[i, 0];
            float y = reference[i, 1];
            // Scale points if image_size is different
            if (image_size != 112)
            {
                x = x / 112 * image_size;
                y = y / 112 * image_size;
            }
            dstPoints[i] = new Point2f(x, y);
        }

        return dstPoints;
    }
}