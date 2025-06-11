Of course! Here is a well-structured GitHub README in Markdown, based on the C# code you provided. It explains the project's purpose, its connection to the `insightface` library, the technologies used, and the overall processing pipeline.

---

# FaceSwap ONNX for C#

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C#](https://img.shields.io/badge/C%23-11-blueviolet)](https://docs.microsoft.com/en-us/dotnet/csharp/)

High-performance, offline face swapping in C#/.NET using ONNX Runtime.

This project is an unofficial C# port of the core inference logic from the popular Python library **[insightface](https://pypi.org/project/insightface/)**, specifically its face-swapping capabilities. It aims to provide a native .NET solution for developers looking to integrate state-of-the-art face analysis and swapping into their applications without a Python dependency. Crucially, this allows parallelization in dot net, whereas the Python implementation is inherenently limited to single threaded processing owing to the Global Interpretter Lock.

---

## Table of Contents

- [Key Features](#key-features)
- [Core Technologies](#core-technologies)
- [The Inference Pipeline](#the-inference-pipeline)
- [ONNX Models Used](#onnx-models-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage Example](#usage-example)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Key Features

- **Face Detection:** Highly accurate face detection using the `RetinaFace` model.
- **High-Quality Face Swapping:** Leverages the `InSwapper` model to seamlessly replace faces in images and videos.
- **Face Analysis:**
    - Extracts 2D and 3D facial landmarks.
    - Generates face embeddings (`ArcFace`) for recognition and comparison.
    - Predicts gender and age.
- **AI-Powered Face Restoration (Upscaling):** Includes an `ESRGAN` model to enhance the quality and resolution of the swapped face.
- **Video Processing:** Built-in `FfmpegClient` to handle video-to-frame extraction and frame-to-video composition.
- **Optimized for Performance:** Uses `ONNX Runtime` for hardware-accelerated inference and includes model warmup capabilities for low-latency processing.

## Core Technologies

This project is built on a modern .NET stack, leveraging powerful libraries for machine learning and computer vision.

| Technology | Role |
| :--- | :--- |
| **.NET / C#** | The core programming language and runtime for the entire application. |
| **Microsoft.ML.OnnxRuntime** | The primary engine for running inference on the pre-trained ONNX models. It enables cross-platform, high-performance ML. |
| **OpenCvSharp** | An extensive C# wrapper for OpenCV, used for all image manipulation tasks like reading, writing, resizing, transformations, and drawing. |
| **FFmpeg** | Used as an external process for robust and efficient video decoding (splitting into frames) and encoding (joining frames into a video). |

## The Inference Pipeline

The face-swapping process is a multi-stage pipeline where the output of one model becomes the input for the next. The `FaceSwapper` class orchestrates this entire workflow.

1.  **Detect Faces (`RetinaFace`)**: The process starts by analyzing the target image or video frame to find all faces. `RetinaFace` returns bounding boxes and preliminary key points for each detected face.

2.  **Analyze Source Face (`ArcFace`)**: The source image (the face you want to swap *in*) is processed. A unique "signature" or **embedding** is generated for the source face using the `ArcFace` model. This embedding mathematically represents the identity of the person.

3.  **Process Target Faces**: For each face detected in the target media:
    *   **Landmark Extraction (`Landmark2d` / `Landmark3d`)**: More precise facial landmarks are detected to understand the face's orientation and expression.
    *   **Face Alignment (`FaceAlign`)**: The target face is warped and cropped based on its landmarks to match the standardized input size required by the swapping model.

4.  **Swap Faces (`InSwapper`)**: The core swapping operation. The `InSwapper` model takes the aligned target face, the source face's embedding, and generates a new face that combines the source's identity with the target's pose, expression, and lighting.

5.  **Paste and Blend**: The newly generated face is pasted back onto the original target image at the correct location, often with blending to smooth the seams.

6.  **(Optional) Restore Face (`EsrGan`)**: To combat potential quality loss, the swapped face can be passed through the `EsrGan` model to upscale it and restore fine details.

7.  **Compose Video (`FfmpegClient`)**: If processing a video, all the modified frames are sequentially combined back into a final video file.

## ONNX Models Used

This library is designed to work with a specific set of ONNX models, each responsible for a part of the pipeline.

-   `RetinaFace`: Face detection.
-   `ArcFace`: Creates a 512-dimension feature vector (embedding) from a face.
-   `Landmark2d` / `Landmark3d`: Detects key facial points.
-   `GenderAge`: Predicts the gender and age of a detected face.
-   `InSwapper`: The main face-swapping model.
-   `EsrGan`: Super-resolution model for face enhancement.

You must download these models and place them in a designated directory for the application to function.

## Getting Started

### Prerequisites

1.  **.NET SDK**: .NET 6 or newer.
2.  **FFmpeg**: You must have `ffmpeg.exe` installed and accessible in your system's PATH or specify its location in the configuration.
3.  **ONNX Models**: Download the required `.onnx` model files and place them in a folder (e.g., `onnx_models`).

### Usage Example

The following demonstrates a high-level overview of how to use the `FaceSwapper` class.

```csharp
using FaceSwapOnnx;
using OpenCvSharp;

public class Program
{
    public static async Task Main(string[] args)
    {
        // 1. Configure Options
        // Point to your model directory, ffmpeg path, and input/output folders.
        var options = new FaceSwapperOptions
        {
            OnnxModelDir = "path/to/your/onnx_models",
            FfmpegPath = "path/to/your/ffmpeg.exe",
            LocalImageDir = "path/to/images",
            LocalVideoDir = "path/to/videos",
            WarmupInferenceAtStartup = true,
            TrackStatistics = true
        };

        // 2. Initialize Dependencies (using a DI container is recommended)
        var jobStatsCollector = new JobStatsCollector();
        var retinaFace = new RetinaFace(options, jobStatsCollector);
        var arcFace = new Arcface(options, jobStatsCollector);
        var inSwapper = new InSwapper(options, jobStatsCollector);
        // ... initialize all other models and FfmpegClient

        // 3. Create the Main FaceSwapper Service
        var faceSwapper = new FaceSwapper(
            options,
            jobStatsCollector,
            retinaFace,
            // ... pass all other dependencies here
        );

        // 4. Warm up the models for faster first-time inference
        faceSwapper.Initialize();

        // 5. Process an Image
        Mat sourceImage = Cv2.ImRead("path/to/source_face.jpg");
        Mat targetImage = Cv2.ImRead("path/to/target_image.jpg");

        // The SwapAsync method would encapsulate the entire pipeline
        Mat resultImage = await faceSwapper.SwapAsync(sourceImage, targetImage);
        
        Cv2.ImWrite("path/to/result.jpg", resultImage);
        Console.WriteLine("Face swap complete!");
    }
}
```

## Configuration

The behavior of the library is controlled via the `FaceSwapperOptions` class. Here you can specify:
-   Paths to your models, FFmpeg, and working directories.
-   Execution providers for ONNX Runtime (e.g., `Cpu`, `Cuda`, `Dml`).
-   Toggles for performance features like model warmup and statistics tracking.

## Acknowledgments

-   This project would not be possible without the incredible research and open-source contributions of the **insightface** team. A huge thank you to them for their work in the field of face analysis. Please visit their original repository: [deepinsight/insightface](https://github.com/deepinsight/insightface).

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
