using System.Drawing;

namespace FaceSwapOnnx.Outputs;

public class KeyFaceFeatures
{
    public double[] PointsFlattened { get; set; }
    public PointF LeftEye { get; set; }
    public PointF RightEye { get; set; }
    public PointF Nose { get; set; }
    public PointF LeftMouthCorner { get; set; }
    public PointF RightMouthCorner { get; set; }

    public KeyFaceFeatures(double[] setOf10Doubles)
    {
        if (setOf10Doubles.Length != 10) throw new ArgumentException("Expected an array of 10 doubles.", nameof(setOf10Doubles));

        PointsFlattened = setOf10Doubles;

        var points = new[]
        {
            new PointF((float)setOf10Doubles[0], (float)setOf10Doubles[1]),
            new PointF((float)setOf10Doubles[2], (float)setOf10Doubles[3]),
            new PointF((float)setOf10Doubles[4], (float)setOf10Doubles[5]),
            new PointF((float)setOf10Doubles[6], (float)setOf10Doubles[7]),
            new PointF((float)setOf10Doubles[8], (float)setOf10Doubles[9])
        };

        LeftEye = points[0];
        RightEye = points[1];
        Nose = points[2];
        LeftMouthCorner = points[3];
        RightMouthCorner = points[4];
    }
}
