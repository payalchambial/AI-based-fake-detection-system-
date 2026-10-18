

Project Overview

DeepGuard AI is an AI-based fake video detection system designed to analyze video frames and identify potential deepfake content using a pre-trained Machine Learning model. The system provides a preliminary classification of videos as Real or Fake through a simple, interactive web interface.

 Features

🎥 Upload videos in MP4, AVI, MOV, and MKV formats.

🤖 AI-powered video frame classification.

🔍 Analyze up to 8 sampled frames from a video.

📊 Display Real and Fake prediction scores.

⚠️ Generate a preliminary decision based on model predictions.

📄 Download a text report containing detection results.

🌐 User-friendly web interface built with Streamlit.

🛠️ Technologies Used

Programming Language: Python

Web Framework: Streamlit

AI/ML: Hugging Face Transformers

Pre-trained Model: dima806/deepfake_vs_real_image_detection

Video Processing: OpenCV

Data Handling: Pandas

Deep Learning Framework: PyTorch

⚙️ How It Works

The user uploads a video file.

The system extracts selected frames from the video.

Each sampled frame is analyzed using a pre-trained AI image-classification model.

The model generates Real and Fake prediction scores for the analyzed frames.

The system aggregates the frame-level results to produce a preliminary decision.

The user can view the results and download a text report.

🎯 Project Objective

The objective of this project is to explore how Artificial Intelligence and Machine Learning can assist in identifying potentially manipulated video content and help users assess digital media authenticity.
