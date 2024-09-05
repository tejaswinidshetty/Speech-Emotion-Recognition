# Speech Emotion Recognition

This project is designed to recognize emotions from speech recordings using machine learning techniques. It utilizes a Random Forest classifier to predict emotions based on audio features extracted from recorded audio data.

## Features

- **Audio Feature Extraction:** Extracts MFCCs, chroma features, and spectral contrast from audio files.
- **Data Augmentation:** Enhances the dataset by applying pitch shifting, time stretching, and adding noise to audio recordings.
- **SMOTE:** Applies Synthetic Minority Over-sampling Technique to handle class imbalance.
- **Random Forest Classifier:** Trains and evaluates a Random Forest model to classify emotions.
- **Real-Time Emotion Prediction:** Records audio and predicts the emotion in real-time using the trained model.

## Requirements

- `numpy`
- `librosa`
- `soundfile`
- `scikit-learn`
- `imblearn`
- `sounddevice`
- `joblib`

You can install the required packages using pip:

```bash
pip install numpy librosa soundfile scikit-learn imbalanced-learn sounddevice joblib
