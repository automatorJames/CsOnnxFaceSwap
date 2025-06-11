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

/// <summary>
/// Generates an embedding, i.e. a "signature" for the face encoding its key features
/// </summary>
public class Arcface(FaceSwapperOptions options, JobStatsCollector jobStatsCollector) : OnnxModelBase(GetModelConfig(options), jobStatsCollector)
{
    static ModelConfig GetModelConfig(FaceSwapperOptions options) => new ModelConfig(
            OnnxModel.Arcface,
            options,
            inputMean: 127.5,
            inputStandard: 127.5
        );

    public double[] Process(Mat img, Face face)
    {
        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobStart(face.JobId, face.FaceId, GetType());

        // Convert face.kps (double[]) to List<PointF>
        double[] kpsArray = face.KeyFaceFeatures.PointsFlattened;

        // Calculate the number of keypoints
        int numKeypoints = kpsArray.Length / 2;

        // Initialize a Mat with dimensions (numKeypoints x 2) and type CV_32F
        Mat kps = new Mat(numKeypoints, 2, MatType.CV_64F);

        // Populate the Mat
        for (int i = 0; i < numKeypoints; i++)
        {
            kps.Set(i, 0, kpsArray[i * 2]);       // X-coordinate
            kps.Set(i, 1, kpsArray[i * 2 + 1]);   // Y-coordinate
        }

        // todo: just use floats
        kps.ConvertTo(kps, MatType.CV_32FC1);

        // Perform face alignment
        Mat aimg = FaceAlign.NormCrop(img, kps, _modelConfig.InputSize.Item1);

        // Get the embedding
        double[] embedding = GetFeat(aimg);

        face.Embedding = embedding;

        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobEnd(face.JobId, face.FaceId, GetType());

        return embedding;
    }

    public double[] GetFeat(Mat img)
    {
        // Create blob from image
        Mat blob = CvDnn.BlobFromImage(
            image: img,
            scaleFactor: 1.0 / _modelConfig.InputStandard,
            size: new Size(_modelConfig.InputSize.Item1, _modelConfig.InputSize.Item2),
            mean: new Scalar(_modelConfig.InputMean, _modelConfig.InputMean, _modelConfig.InputMean),
            swapRB: true,
            crop: false);

        // Extract blob data
        int totalSize = (int)blob.Total();
        float[] blobData = new float[totalSize];
        Marshal.Copy(blob.Data, blobData, 0, totalSize);

        var inputTensor = blob.ToDenseTensor();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_modelConfig.InputNames[0], inputTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var net_out = results.First().AsTensor<float>();

        float[] net_out_array = net_out.ToArray();

        double[] net_out_double = Array.ConvertAll(net_out_array, x => (double)x);

        return net_out_double;
    }
}