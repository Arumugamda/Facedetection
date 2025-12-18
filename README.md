# Facedetection
## Face Recognition System using OpenCV & LBPH

Abishek Pranav G V | Student ID: 22WU0204006 |
abishekpranav403@gmail.com

## Project Overview

This project implements a complete real-time Face Recognition System using classical computer vision techniques. It covers the full pipeline — from dataset creation to model training and live recognition — built with Python and OpenCV.

The system combines:

>Haar Cascade Classifiers for face detection
>Local Binary Patterns Histograms (LBPH) for face recognition

The result is a lightweight, fast, and interpretable biometric system that works well under controlled conditions.

Abishek Pranav G V | Student ID: 22WU0204006 |
abishekpranav403@gmail.com

# System Architecture

The project is structured into three modular Python scripts:
    1.Dataset Generation
Captures and stores face images using a webcam.
    2.Model Training
Trains an LBPH face recognizer on the collected dataset.
    3.Real-Time Recognition
Detects and recognizes faces from a live video stream.
    This separation keeps the system clean, extensible, and easy to debug.

# Workflow

1.Webcam captures live video
2.Frames are converted to grayscale
3.Faces are detected using Haar Cascades
4.Detected faces are passed to the LBPH model
5.Output displays:
    >Bounding box
    >User name
    >Recognition confidence

# Implementation Details
## Dataset Generation
>Captures up to 500 images per user
>Faces are cropped and saved in grayscale
>Naming format:
    user.<ID>.<image_number>.jpg
>Live visual feedback during capture
>Basic error handling for camera and file issues
        This step builds the raw memory of the system.

# Model Training

    > Reads all face images from the dataset folder
    > Extracts user IDs directly from filenames
    > Trains an LBPH Face Recognizer
    > Saves trained model as classifier.yml

Why LBPH?
    > Works well with small datasets
    > Robust to lighting variation
    > Computationally efficient

Old-school, but dependable.

# Face Recognition
>Uses trained LBPH model
>Maps user IDs to names using a dictionary
>Displays confidence as:
100 - raw_confidence
>Threshold:
    Confidence < 50 → Known face
    Confidence ≥ 50 → Unknown
Real-time. No fluff.

# Technical Deep Dive
Haar Cascade Face Detection
>Uses OpenCV’s pre-trained frontal face classifier
>Parameters:
    scaleFactor = 1.1
    minNeighbors = 10
>Fast and lightweight
>Best for frontal faces under decent lighting
Classic tech. Still relevant.

# LBPH Face Recognition
    Divides face into grids
    Converts pixel neighborhoods into binary patterns
    Builds histograms for each region
    Compares histograms using distance metrics

Lower confidence score = better match.

Simple math. Solid results.

# Performance Analysis
## Detection Accuracy
    Strong under normal lighting
    Best with frontal faces
    Weak with extreme angles or shadows
## Recognition Confidence
    Intuitive confidence display
    Reliable with good-quality training data
## Limitations
    Sensitive to lighting and facial accessories
    Struggles with aging and expression changes
    Fixed confidence threshold may need tuning

No illusions. Just facts.

# Applications
    Access control systems
    Automated attendance
    Smart device personalization
    Retail customer recognition

Anywhere identity meets automation.

## Possible Extensions

    Deep learning models (CNNs, FaceNet)
    Liveness detection (anti-spoofing)
    Multi-camera support
    Database-backed user management
    GUI for configuration and monitoring

This is a foundation, not a finish line.

Conclusion

This project proves that classical computer vision still matters. By combining Haar Cascades and LBPH, the system achieves a practical balance between accuracy, speed, and simplicity.

The modular design allows easy upgrades, making it a strong base for more advanced facial recognition systems. It respects the past, works in the present, and leaves room for smarter futures.

References

OpenCV Documentation – Haar Cascade Face Detection

OpenCV Documentation – LBPH Face Recognition

Ahonen, T., Hadid, A., & Pietikäinen, M. (2006).
Face Description with Local Binary Patterns. IEEE TPAMI.


