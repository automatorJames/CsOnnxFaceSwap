using System.Drawing;

namespace FaceSwapOnnx.Outputs;

public class FacePose
{
    public double Pitch { get; set; }
    public double Yaw { get; set; }
    public double Roll { get; set; }
    public double DeviationFromIdeal { get; set; }
    public bool MeetsIdealThreshold { get; set; }

    public FacePose(double pitchX, double yawY, double rollZ)
    {
        Pitch = pitchX;
        Yaw = yawY;
        Roll = rollZ;

        double weightedThreshold = 0.2;

        double pitchMax = 30;
        double yawMax = 35;
        double rollMax = 40;

        double pitchAbs = Math.Abs(Pitch);
        double yawAbs = Math.Abs(Yaw);
        double rollAbs = Math.Abs(Roll);

        double pitchWeight = 0.45;
        double yawWeight = 0.30;
        double rollWeight = 0.30;

        var pitchNormalized = pitchAbs / 180;
        var yawNormalized = yawAbs / 180;
        var rollNormalized = rollAbs / 180;

        var pitchWeghted = pitchNormalized * pitchWeight;
        var yawWeghted = yawNormalized * yawWeight;
        var rollWeghted = rollNormalized * rollWeight;

        var weightedScore = pitchWeghted + yawWeghted + rollWeghted;
        DeviationFromIdeal = Math.Min(1, weightedScore);

        MeetsIdealThreshold = DeviationFromIdeal < weightedThreshold;
        MeetsIdealThreshold &= pitchAbs < pitchMax && yawAbs < yawMax && rollAbs < rollMax;
    }

    public string GetFaceOverlayTextForSite() => Pitch.ToString("F0") + "Y|" + Yaw.ToString("F0") + "°\n" + "P|" + "°\n" + "R|" + Roll.ToString("F0") + "°";

    public override string ToString() => "P|" + Pitch.ToString("F0") + "° " + "Y|" + Yaw.ToString("F0") + "° " + "R|" + Roll.ToString("F0") + "°";
}
