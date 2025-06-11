namespace FaceSwapOnnx.Models;

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using FaceSwapOnnx.Outputs;
using FaceSwapOnnx.Statistics;

public class GenderAge(FaceSwapperOptions options, JobStatsCollector jobStatsCollector) : OnnxModelBase(GetModelConfig(options), jobStatsCollector)
{
    static ModelConfig GetModelConfig(FaceSwapperOptions options) => new ModelConfig(
            OnnxModel.GenderAge,
            options,
            inputMean: 0.0,
            inputStandard: 1.0
        );

    public object Process(Mat img, Face face)
    {
        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobStart(face.JobId, face.FaceId, GetType());

        // Extract bbox and compute center, scale, rotate
        double[] bbox = face.BoundingBox.PointsFlattened; // [x1, y1, x2, y2]
        double w = bbox[2] - bbox[0];
        double h = bbox[3] - bbox[1];
        //double[] centerCoords = { (bbox[2] + bbox[0]) / 2.0f, (bbox[3] + bbox[1]) / 2.0f };
        var center = Tuple.Create((bbox[2] + bbox[0]) / 2.0f, (bbox[3] + bbox[1]) / 2.0f);
        double rotate = 0;
        double _scale = _modelConfig.InputSize.Item1 / (Math.Max(w, h) * 1.5f);


        // Align the face
        Mat aimg, M;
        FaceAlign.Transform(img, center, _modelConfig.InputSize.Item1, _scale, rotate, out aimg, out M);

        // Prepare the blob for input
        Mat blob = CvDnn.BlobFromImage(
            aimg,
            1.0 / _modelConfig.InputStandard,
            new Size(_modelConfig.InputSize.Item1, _modelConfig.InputSize.Item2),
            new Scalar(_modelConfig.InputMean, _modelConfig.InputMean, _modelConfig.InputMean),
            swapRB: true);

        var inputTensor = blob.ToDenseTensor();

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_modelConfig.InputNames[0], inputTensor)
        };

        using var results = _session.Run(inputs);
        var predTensor = results.First().AsTensor<float>();
        float[] predData = predTensor.ToArray();

        if (predData.Length != 3)
            throw new Exception("Expected prediction of length 3 for genderage task");

        // Gender is argmax of predData[0:2]
        int gender = predData[0] > predData[1] ? 0 : 1;

        // Age is predData[2] * 100, rounded to int
        int age = (int)Math.Round(predData[2] * 100);

        face.Gender = (Gender)gender;
        face.Age = age;

        if (_options.TrackStatistics)
            _jobStatsCollector.SignalJobEnd(face.JobId, face.FaceId, GetType());

        return new Tuple<int, int>(gender, age);
    }
}