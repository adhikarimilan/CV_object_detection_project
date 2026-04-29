# Project: Computer Vision Vehicle Counter

## Project Overview

This project implements an automated vehicle detection and counting system using computer vision. It processes video feeds to identify, track, and count vehicles crossing a specific boundary in real time.

## Technical Stack

- **Language**: Python
- **Object Detection**: YOLOv8 (You Only Look Once)
- **Object Tracking**: SORT (Simple Online and Realtime Tracking)
- **Computer Vision Libraries**: OpenCV, cvzone
- **Data Processing**: NumPy

## Core Functionalities

- **Real-Time Detection**: Utilizes a pre-trained YOLOv8 model to identify vehicle classes including cars, trucks, buses, and motorcycles.
- **Unique Object Tracking**: Integrates the SORT algorithm to assign and maintain unique IDs for every detected vehicle, preventing double counting.
- **Image Masking**: Employs a custom binary mask to define a specific Region of Interest (ROI), focusing the algorithm on the roadway and reducing background noise.
- **Dynamic Counting Logic**: Features a coordinate-based trigger line. The system increments the counter only when a vehicle's center point (centroid) intersects the defined boundary.
- **Visual Feedback**: Provides a live video overlay with bounding boxes, tracking IDs, confidence scores, and a persistent counter.

## Implementation Details

- **Filtering**: The script specifically filters for road-based COCO classes to ensure accuracy.
- **Efficiency**: Uses stream-based processing to maintain high frame rates during detection.
- **Robustness**: The SORT tracker is configured with parameters for maximum age and minimum hits to handle temporary occlusions.

## Impact

This tool can be applied to traffic flow analysis, urban planning, and automated toll systems. It demonstrates proficiency in deploying state-of-the-art deep learning models for practical monitoring solutions.

## Screenshots

![Demo1](/resources/image/demo/1.png "This is a demo 1 image.")
![Demo2](/resources/image/demo/2.png "This is a demo 2 image.")
![Demo3](/resources/image/demo/3.png "This is a demo 3 image.")
![Demo4](/resources/image/demo/4.png "This is a demo 4 image.")
