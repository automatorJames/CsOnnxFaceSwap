using System.Drawing;

namespace FaceSwapOnnx.Outputs;

public class BoundingBox
{
    public int ImageHeight { get; set; }
    public int ImageWidth { get; set; }
    public double[] PointsFlattened { get; set; }
    public RectangleF Rectangle { get; set; }
    public RectangleF PaddedRectangle { get => GetPaddedRectangle(); }

    public float Area { get => Rectangle.Height * Rectangle.Width; }

    public BoundingBox(double[] setOf4Doubles, Mat image)
    {
        PointsFlattened = setOf4Doubles;
        var topLeft = new PointF((float)setOf4Doubles[0], (float)setOf4Doubles[1]);
        var bottomRight = new PointF((float)setOf4Doubles[2], (float)setOf4Doubles[3]);
        Rectangle = new RectangleF(topLeft.X, topLeft.Y, bottomRight.X - topLeft.X, bottomRight.Y - topLeft.Y);
        ImageHeight = image.Height;
        ImageWidth = image.Width;
    }

    RectangleF GetPaddedRectangle(float paddingPct = .33f)
    {
        // Calculate expansion values
        float expansionHeight = paddingPct * ImageHeight;
        float expansionWidth = paddingPct * ImageWidth;

        expansionHeight = Math.Min(expansionHeight, ImageHeight);
        expansionWidth = Math.Min(expansionWidth, ImageWidth);

        var desiredX = Rectangle.X - expansionWidth / 2;
        var desiredY = Rectangle.Y - expansionHeight / 2;
        var desiredWidth = Rectangle.Width + expansionWidth;
        var desiredHeight = Rectangle.Height + expansionHeight;

        var newX = Math.Max(0, desiredX);
        var newY = Math.Max(0, desiredY);
        var newWidth = Math.Min(ImageWidth - newX, desiredWidth);
        var newHeight = Math.Min(ImageHeight - newY, desiredHeight);

        // Create and return the expanded rectangle
        RectangleF expandedRect = new RectangleF(newX, newY, newWidth, newHeight);

        return expandedRect;
    }
}
