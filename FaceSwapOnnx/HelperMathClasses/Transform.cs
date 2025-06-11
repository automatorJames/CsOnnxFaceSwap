namespace FaceSwapOnnx.HelperMathClasses;

using OpenCvSharp;
using System;

public static class TransformHelpers
{
    public static Mat TransPoints(Mat pts, Mat M)
    {
        if (pts.Channels() == 2)
            return TransPoints2D(pts, M);
        else if (pts.Channels() == 3)
            return TransPoints3D(pts, M);
        else
            throw new ArgumentException("Unsupported number of channels. Mat must have 2 or 3 channels.");
    }

    public static Mat TransPoints2D(Mat pts, Mat M)
    {
        if (pts.Channels() != 2)
            throw new ArgumentException("Expected a Mat with 2 channels for 2D points.");

        Mat newPts = new Mat(pts.Rows, pts.Cols, MatType.CV_32FC2);
        for (int i = 0; i < pts.Rows; i++)
        {
            Vec2f pt = pts.At<Vec2f>(i, 0);

            // Create a Mat for the homogeneous point
            Mat ptMat = new Mat(3, 1, MatType.CV_32F);
            ptMat.Set(0, 0, pt.Item0);
            ptMat.Set(1, 0, pt.Item1);
            ptMat.Set(2, 0, 1.0f);

            // Multiply M with the homogeneous point
            Mat transformedPt = M * ptMat;

            // Extract the transformed point and store it
            newPts.Set(i, 0, new Vec2f(
                transformedPt.At<float>(0, 0),
                transformedPt.At<float>(1, 0)
            ));
        }

        return newPts;
    }

    public static Mat TransPoints3D(Mat pts, Mat M)
    {
        if (pts.Channels() != 3)
            throw new ArgumentException("Expected a Mat with 3 channels for 3D points.");

        Mat newPts = new Mat(pts.Rows, pts.Cols, MatType.CV_32FC3);

        // Compute scaling factor
        float scale = (float)Math.Sqrt(Math.Pow(M.At<float>(0, 0), 2) + Math.Pow(M.At<float>(0, 1), 2));

        for (int i = 0; i < pts.Rows; i++)
        {
            Vec3f pt = pts.At<Vec3f>(i, 0);

            // Create a Mat for the homogeneous point
            Mat ptMat = new Mat(3, 1, MatType.CV_32F);
            ptMat.Set(0, 0, pt.Item0);
            ptMat.Set(1, 0, pt.Item1);
            ptMat.Set(2, 0, 1.0f);

            // Multiply M with the homogeneous point
            Mat transformedPt = M * ptMat;

            // Extract the transformed point and scale the Z coordinate
            newPts.Set(i, 0, new Vec3f(
                transformedPt.At<float>(0, 0),
                transformedPt.At<float>(1, 0),
                pt.Item2 * scale
            ));
        }

        return newPts;
    }



    public static Mat EstimateAffineMatrix3D23D(Mat X, Mat Y)
    {

        int n = X.Rows; // Number of points

        // Append a column of ones to X to make it homogeneous
        Mat ones = Mat.Ones(n, 1, type: MatType.CV_32F);
        Mat X_homo = new Mat();
        Cv2.HConcat(new Mat[] { X, ones }, X_homo); // X_homo: (n x 4)

        // Solve for P in the equation X_homo * P.T = Y
        // Using least squares solution: P_T = (X_homo_inv * Y)
        // Since X_homo is not square, use SVD or pseudo-inverse

        Mat P_T = new Mat();
        Cv2.Solve(X_homo, Y, P_T, DecompTypes.QR); // P_T: (4 x 3)

        // Transpose P_T to get P: (3 x 4)
        Mat P = P_T.T();

        return P;
    }


    public static (double s, Mat R, Mat t) P2sRt(Mat P)
    {
        // Validate input
        if (P.Rows != 3 || P.Cols != 4)
        {
            throw new ArgumentException("P must be of shape (3 x 4).");
        }

        /*        // Ensure P is of type CV_64F
                if (P.Type() != MatType.CV_64F)
                    P = P.Clone().ConvertTo(MatType.CV_64F);*/

        // Extract translation vector t (last column of P)
        Mat t = P.Col(3).Clone(); // t: (3 x 1)

        // Extract R1 and R2 (first two rows of P, excluding last column)
        Mat R1 = P.Row(0).ColRange(0, 3).Clone(); // R1: (1 x 3)
        Mat R2 = P.Row(1).ColRange(0, 3).Clone(); // R2: (1 x 3)

        // Compute scale factor s
        double normR1 = Cv2.Norm(R1);
        double normR2 = Cv2.Norm(R2);
        double s = (normR1 + normR2) / 2.0;

        // Normalize R1 and R2 to get r1 and r2
        Mat r1 = R1 / normR1; // r1: (1 x 3)
        Mat r2 = R2 / normR2; // r2: (1 x 3)

        // Convert r1 and r2 to Vec3d
        Vec3d r1Vec = new Vec3d(r1.At<float>(0, 0), r1.At<float>(0, 1), r1.At<float>(0, 2));
        Vec3d r2Vec = new Vec3d(r2.At<float>(0, 0), r2.At<float>(0, 1), r2.At<float>(0, 2));

        // Compute r3 as the cross product of r1 and r2
        Vec3d r3Vec = CrossProduct(r1Vec, r2Vec);

        // Construct rotation matrix R
        Mat R = new Mat(3, 3, MatType.CV_64F);
        R.Set(0, 0, r1Vec.Item0); R.Set(0, 1, r1Vec.Item1); R.Set(0, 2, r1Vec.Item2);
        R.Set(1, 0, r2Vec.Item0); R.Set(1, 1, r2Vec.Item1); R.Set(1, 2, r2Vec.Item2);
        R.Set(2, 0, r3Vec.Item0); R.Set(2, 1, r3Vec.Item1); R.Set(2, 2, r3Vec.Item2);

        // todo: just fix to work with floats natively?
        R.ConvertTo(R, MatType.CV_32FC1);

        return (s, R, t);
    }

    public static Vec3d CrossProduct(Vec3d a, Vec3d b)
    {
        double cx = a.Item1 * b.Item2 - a.Item2 * b.Item1;
        double cy = a.Item2 * b.Item0 - a.Item0 * b.Item2;
        double cz = a.Item0 * b.Item1 - a.Item1 * b.Item0;
        return new Vec3d(cx, cy, cz);
    }

    public static (double x, double y, double z) Matrix2Angle(Mat R)
    {
        // Validate input
        if (R.Rows != 3 || R.Cols != 3)
        {
            throw new ArgumentException("R must be of shape (3 x 3).");
        }

        /*        // Ensure R is of type CV_64F
                if (R.Type() != MatType.CV_64F)
                    R = R.Clone().ConvertTo(MatType.CV_64F);*/

        double sy = Math.Sqrt(R.At<float>(0, 0) * R.At<float>(0, 0) + R.At<float>(1, 0) * R.At<float>(1, 0));

        bool singular = sy < 1e-6;

        double x, y, z;

        if (!singular)
        {
            x = Math.Atan2(R.At<float>(2, 1), R.At<float>(2, 2)); // Pitch
            y = Math.Atan2(-R.At<float>(2, 0), sy);                // Yaw
            z = Math.Atan2(R.At<float>(1, 0), R.At<float>(0, 0)); // Roll
        }
        else
        {
            x = Math.Atan2(-R.At<float>(1, 2), R.At<float>(1, 1)); // Pitch
            y = Math.Atan2(-R.At<float>(2, 0), sy);                 // Yaw
            z = 0;
        }

        // Convert from radians to degrees
        double rx = x * (180.0 / Math.PI); // Pitch
        double ry = y * (180.0 / Math.PI); // Yaw
        double rz = z * (180.0 / Math.PI); // Roll

        return (rx, ry, rz);
    }
}
