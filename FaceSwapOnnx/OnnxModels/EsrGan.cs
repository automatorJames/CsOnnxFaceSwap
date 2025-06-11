namespace FaceSwapOnnx.Models;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;
using FaceSwapOnnx.Outputs;
using FaceSwapOnnx.Statistics;

public class EsrGan(FaceSwapperOptions options, JobStatsCollector jobStatsCollector) : OnnxModelBase(GetModelConfig(options), jobStatsCollector)
{
    static ModelConfig GetModelConfig(FaceSwapperOptions options) => new ModelConfig(
            OnnxModel.EsrGan,
            options,
            inputMean: 0,
            inputStandard: 0
        );

    public Mat Upscale(Mat image, Face face)
    {
        int h = image.Rows, w = image.Cols;

        // ESRGAN expects floats in [0,1]
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, h, w });
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Vec3b pix = image.At<Vec3b>(y, x);
                // Channel order: R=0, G=1, B=2
                inputTensor[0, 0, y, x] = pix.Item0 / 255f;
                inputTensor[0, 1, y, x] = pix.Item1 / 255f;
                inputTensor[0, 2, y, x] = pix.Item2 / 255f;
            }
        }

        // Create NamedOnnxValue for the first (and usually only) input
        string inputName = _session.InputMetadata.Keys.First();
        string outputName = _session.OutputMetadata.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        // Run inference
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

        // Extract output tensor
        var outputTensor = results.First(x => x.Name == outputName).AsTensor<float>();

        var dims = outputTensor.Dimensions;

        // dims = [1, 3, h*scale, w*scale]
        int oh = dims[2], ow = dims[3];

        // Convert back to byte image
        Mat outBgr = new Mat(oh, ow, MatType.CV_8UC3);
        for (int y = 0; y < oh; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                // clamp and convert to [0–255]
                byte r = (byte)Math.Min(255, Math.Max(0, (int)(outputTensor[0, 0, y, x] * 255)));
                byte g = (byte)Math.Min(255, Math.Max(0, (int)(outputTensor[0, 1, y, x] * 255)));
                byte b = (byte)Math.Min(255, Math.Max(0, (int)(outputTensor[0, 2, y, x] * 255)));
                outBgr.Set(y, x, new Vec3b(b, g, r));
            }
        }

        return outBgr;
    }
}