namespace FaceSwapOnnx.Outputs;

public class Face
{
    public Guid JobId { get; set; }
    public Guid FaceId { get; set; }
    public int ImageHeight { get; set; }
    public int ImageWidth { get; set; }
    public double DetectionScore { get; set; }
    public double[] Embedding { get; set; }
    public double EmbeddingNorm { get => Norm(Embedding.ToArray()); }
    public BoundingBox BoundingBox { get; private set; }
    public KeyFaceFeatures KeyFaceFeatures { get; set; }
    public FacePose Pose { get; set; }
    public Gender Gender { get; set; }
    public int Age { get; set; }
    public int OrderLeftToRight { get; set; }
    public double[] EmbeddingSignature { get; set; }
    public dynamic SwappedImgData { get; set; }
    public string SourceFileName { get; set; }
    public int ImageArea => ImageHeight * ImageWidth;
    public double FaceSizePercentage  => BoundingBox.Area / ImageArea;

    public Mat GetNormedEmbedding()
    {
        Mat embeddingMat = new Mat(Embedding.Length, 1, MatType.CV_64F);

        for (int i = 0; i < Embedding.Length; i++)
            embeddingMat.Set(i, 0, Embedding[i]);

        Mat normedMat = new Mat();
        Cv2.Divide(embeddingMat, EmbeddingNorm, normedMat);

        return normedMat;
    }

    public Face(Guid jobId, Guid faceId, double[] boundingBoxPoints, double[] faceDetectionKeyPoints, double detectionScore, Mat image)
    {
        ImageHeight = image.Height;
        ImageWidth = image.Width;
        JobId = jobId;
        FaceId = faceId;
        BoundingBox = new BoundingBox(boundingBoxPoints, image);
        KeyFaceFeatures = new KeyFaceFeatures(faceDetectionKeyPoints);
        DetectionScore = detectionScore;
    }

    public static List<Face> FromRetinaFaceResult(RetinaFaceResult retinaFaceResult)
    {
        List<Face> facesUnordered = new();
        List<Face> facesOrdered = new();

        for (int i = 0; i < retinaFaceResult.BoundingBoxesAndDetectionScores.Rows; i++)
        {
            var bboxAndDetScoreRow = retinaFaceResult.BoundingBoxesAndDetectionScores.Row(i);
            var bboxAndDetScore = bboxAndDetScoreRow.MatToFlat();
            var boundingBox = bboxAndDetScore[0..4];
            var detectionScore = bboxAndDetScore[4];
            var keyPointsRow = retinaFaceResult.FaceDetectionKeyPoints.Row(i);
            var keyPointSet = keyPointsRow.MatToFlat();
            var face = new Face(retinaFaceResult.JobId, retinaFaceResult.FaceId, boundingBox, keyPointSet, detectionScore, retinaFaceResult.Image);
            facesUnordered.Add(face);
        }

        int orderLeftToRight = 0;
        foreach (var face in facesUnordered.OrderBy(x => x.BoundingBox.Rectangle.Left))
        {
            face.OrderLeftToRight = orderLeftToRight++;
            facesOrdered.Add(face);
        }

        return facesOrdered;
    }

    public double GetSimilarityScore(Face faceToCompare)
    {
        var embeddingToCompare = faceToCompare.EmbeddingSignature;

        if (EmbeddingSignature.Length != embeddingToCompare.Length)
            throw new ArgumentException("Embeddings must have the same dimensionality.");

        double dotProduct = 0;
        double normEmbedding1 = 0;
        double normEmbedding2 = 0;

        for (int i = 0; i < EmbeddingSignature.Length; i++)
        {
            dotProduct += EmbeddingSignature[i] * embeddingToCompare[i];
            normEmbedding1 += Math.Pow(EmbeddingSignature[i], 2);
            normEmbedding2 += Math.Pow(embeddingToCompare[i], 2);
        }

        normEmbedding1 = Math.Sqrt(normEmbedding1);
        normEmbedding2 = Math.Sqrt(normEmbedding2);

        if (normEmbedding1 == 0 || normEmbedding2 == 0)
            return 0.0;

        // Calculate cosine similarity
        double cosineSimilarity = dotProduct / (normEmbedding1 * normEmbedding2);

        // Convert cosine similarity to a similarity score between 0 and 1
        // 1 indicates identical embeddings, 0 indicates completely dissimilar embeddings
        double similarityScore = 0.5 + cosineSimilarity * 0.5;

        return similarityScore;
    }
}
