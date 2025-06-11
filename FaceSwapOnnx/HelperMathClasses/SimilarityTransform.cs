namespace FaceSwapOnnx.HelperMathClasses;

using OpenCvSharp;
using System;

public class SimilarityTransform
{
    public Mat Params { get; private set; }

    public bool Estimate(Mat src, Mat dst, bool estimateScale = true)
    {
        // Validate inputs
        if (src.Rows != dst.Rows || src.Cols != dst.Cols)
            throw new ArgumentException("Source and destination points must have the same dimensions.");

        if (src.Rows < 2)
            throw new ArgumentException("At least two points are required for estimation.");

        // Call _umeyama to calculate the transformation matrix
        Params = Umeyama(src, dst, estimateScale);

        // Check if Params contains NaN values
        return !HasNaN(Params);
    }

    private Mat Umeyama(Mat src, Mat dst, bool estimateScale)
    {
        int num = src.Rows; // Number of points
        int dim = src.Cols; // Dimensionality of points

        // Compute mean of src and dst
        Mat srcMean = ComputeMean(src);
        Mat dstMean = ComputeMean(dst);

        // Subtract mean from src and dst
        Mat srcDemean = Demean(src, srcMean);
        Mat dstDemean = Demean(dst, dstMean);

        // Compute the covariance matrix
        Mat A = dstDemean.T() * srcDemean / num;

        // Handle determinant sign
        Mat d = Mat.Ones(new Size(1, dim), MatType.CV_32F);
        if (Cv2.Determinant(A) < 0)
            d.Set(dim - 1, 0, -1);

        // Initialize the transformation matrix
        Mat T = Mat.Eye(dim + 1, dim + 1, MatType.CV_32F);

        // Perform Singular Value Decomposition (SVD)
        Mat S = new Mat();
        Mat U = new Mat();
        Mat vt = new Mat();

        Cv2.SVDecomp(A, S, U, vt);

        InvertSecondColumn(U);
        InvertSecondRow(vt);

        // Handle rank
        int rank = ComputeRank(A);
        if (rank == 0)
        {
            T.SetTo(double.NaN);
            return T;
        }
        else if (rank == dim - 1)
        {
            if (Cv2.Determinant(U) * Cv2.Determinant(vt) > 0)
            {
                T[Rect.FromLTRB(0, 0, dim, dim)] = U * vt;
            }
            else
            {
                double lastD = d.At<double>(dim - 1, 0);
                d.Set(dim - 1, 0, -1);
                T[Rect.FromLTRB(0, 0, dim, dim)] = U * DiagonalMatrix(d) * vt;
                d.Set(dim - 1, 0, lastD);
            }
        }
        else
        {
            T[Rect.FromLTRB(0, 0, dim, dim)] = U * DiagonalMatrix(d) * vt;
        }

        // Estimate scale if needed
        double scale = 1.0;
        if (estimateScale)
        {
            double srcVar = Variance(srcDemean);
            Mat product = S.T() * d; // Compute the matrix product
            scale = Cv2.Sum(product).Val0 / srcVar; // Sum the elements of the resulting Mat
        }

        /*        // Compute translation
                Mat scaledTransform = T[Rect.FromLTRB(0, 0, dim, dim)] * scale;
                T[Rect.FromLTRB(0, dim, dim + 1, dim)] = dstMean - (scaledTransform * srcMean.T());

                // Scale the rotation part of the matrix
                T[Rect.FromLTRB(0, 0, dim, dim)] *= scale;*/

        // Compute translation
        Mat scaledTransform = T[Rect.FromLTRB(0, 0, dim, dim)] * scale;

        /*        // Replicate srcMean.T() to match the dimensionality of scaledTransform
                Mat replicatedSrcMean = new Mat(2, 2, MatType.CV_32FC1);
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        replicatedSrcMean.Set<float>(i, j, srcMean.At<float>(j)); // Replicate the values
                    }
                }*/

        //srcMean = DuplicateRows(srcMean, 2);
        //dstMean = DuplicateRows(dstMean, 2);

        Mat T_topLeft = T[Rect.FromLTRB(0, 0, dim, dim)]; // Extract T[:dim, :dim]
        Mat srcMeanTranspose = srcMean.T();              // Transpose src_mean
        Mat prod = T_topLeft * srcMeanTranspose;      // Perform matrix multiplication
        Mat result = prod * scale;                    // Scale the result
        Mat translation = dstMean.T().Subtract(result);

        // Compute translation
        //Mat translation = dstMean - (scaledTransform * srcMean);

        // Adjust translation to be a single column
        //translation = translation.ColRange(0, 1);

        // Assign translation to the correct ROI
        T[Rect.FromLTRB(dim, 0, dim + 1, dim)] = translation;

        // Scale the rotation part of the matrix
        T[Rect.FromLTRB(0, 0, dim, dim)] *= scale;

        //Mat translation = dstMean - (scaledTransform * srcMean);
        //T[Rect.FromLTRB(0, dim, dim + 1, dim)] = translation;

        // Scale the rotation part of the matrix
        //T[Rect.FromLTRB(0, 0, dim, dim)] *= scale;

        return T;
    }

    private Mat ComputeMean(Mat mat)
    {
        Mat mean = new Mat();
        Cv2.Reduce(mat, mean, 0, ReduceTypes.Avg, dtype: -1);
        return mean;
    }

    private Mat Demean(Mat mat, Mat mean)
    {
        // Repeat the mean Mat to match the number of rows in mat
        Mat meanExpanded = new Mat();
        Cv2.Repeat(mean, mat.Rows, 1, meanExpanded);

        // Perform subtraction
        Mat result = new Mat();
        Cv2.Subtract(mat, meanExpanded, result);
        return result;
    }

    private Mat DiagonalMatrix(Mat diag)
    {
        Mat result = Mat.Zeros(diag.Rows, diag.Rows, diag.Type());
        for (int i = 0; i < diag.Rows; i++)
        {
            result.Set(i, i, diag.At<float>(i, 0));
        }
        return result;
    }

    private double Variance(Mat mat)
    {
        Mat square = new Mat();
        Cv2.Multiply(mat, mat, square);
        return Cv2.Sum(square).Val0 / mat.Rows;
    }

    private bool HasNaN(Mat mat)
    {
        // Create a mask where all NaN values are 1 and non-NaN values are 0
        Mat nanMask = new Mat();
        Cv2.Compare(mat, mat, nanMask, CmpType.NE); // NaN != NaN will be true for NaNs

        // Check if the sum of the mask is greater than 0 (indicating NaN exists)
        return Cv2.CountNonZero(nanMask) > 0;
    }

    private static int ComputeRank(Mat mat)
    {
        // Perform Singular Value Decomposition
        Mat w = new Mat();
        Cv2.SVDecomp(mat, w, new Mat(), new Mat());

        // Count singular values greater than the tolerance
        return Cv2.CountNonZero(w);
    }

    // todo: this seems like a hack, but it's necessary b/c numpy's np.linalg.svd treats signs differently than Cv2.SVDecomp
    private void InvertSecondColumn(Mat matrix)
    {
        for (int row = 0; row < matrix.Rows; row++)
        {
            // Invert the second column of the current row
            float value = matrix.At<float>(row, 1);
            matrix.Set(row, 1, -value);
        }
    }

    private void InvertSecondRow(Mat matrix)
    {
        if (matrix.Rows < 2 || matrix.Cols < 2)
        {
            throw new ArgumentException("Matrix must have at least two rows and two columns.");
        }

        for (int col = 0; col < matrix.Cols; col++)
        {
            // Invert the values in the second row
            float value = matrix.At<float>(1, col);
            matrix.Set(1, col, -value);
        }
    }
}