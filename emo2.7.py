import numpy as np
import librosa
import soundfile as sf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import datetime
from joblib import Parallel, delayed

def extract_features(audio, sr=None):
    if isinstance(audio, str):
        audio, sr = librosa.load(audio, res_type='kaiser_fast')

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    mfccs_scaled = np.mean(mfccs.T, axis=0)
    chroma_scaled = np.mean(chroma.T, axis=0)
    contrast_scaled = np.mean(contrast.T, axis=0)

    return np.concatenate([mfccs_scaled, chroma_scaled, contrast_scaled])

def augment_audio(audio, sr):
    pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    time_stretched = librosa.effects.time_stretch(audio, rate=0.8)
    noise = np.random.randn(len(audio))
    audio_with_noise = audio + 0.005 * noise
    return [pitch_shifted, time_stretched, audio_with_noise]

def record_audio(duration=20, fs=22050, output_dir='recordings/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{output_dir}recorded_audio_{current_time}.wav"

    print(f"Recording {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording complete")

    sf.write(output_file, audio.flatten(), fs)
    return output_file

def process_file(file_path, label):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = extract_features(audio, sr)
    augmented_features = []
    augmented_labels = []

    augmented = augment_audio(audio, sr)
    for aug_audio in augmented:
        aug_features = extract_features(aug_audio, sr)
        augmented_features.append(aug_features)
        augmented_labels.append(label)

    return features, augmented_features, augmented_labels

X = []
y = []
ravdess_path = 'A:/speechemo/dataset/archive/all_actor'

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

print("Loading dataset...")
for subdir, dirs, files in os.walk(ravdess_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(subdir, file)
            X.append(file_path)
            label = file.split('-')[2]
            emotion_label = emotion_map[label]
            y.append(emotion_label)

print(f"Total samples loaded: {len(X)}")
print(f"Unique emotions in dataset: {set(y)}")

print("Extracting features and performing data augmentation...")

results = Parallel(n_jobs=-1)(delayed(process_file)(file_path, label) for file_path, label in zip(X, y))

features = []
augmented_features = []
augmented_labels = []

for result in results:
    feature, aug_features, aug_labels = result
    features.append(feature)
    augmented_features.extend(aug_features)
    augmented_labels.extend(aug_labels)

X = features + augmented_features
y = y + augmented_labels

X = np.array(X)
y = np.array(y)

print(f"Total samples after augmentation: {len(X)}")

le = LabelEncoder()
y = le.fit_transform(y)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print("Training and evaluating Random Forest model...")

rf_model = RandomForestClassifier(random_state=42)

rf_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}

rf_search = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_grid, n_iter=200, cv=5, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)

rf_best_model = rf_search.best_estimator_
rf_accuracy = accuracy_score(y_test, rf_best_model.predict(X_test))
print(f'Random Forest Best Accuracy: {rf_accuracy * 100:.2f}%')

def record_and_predict_emotion(duration=20):
    output_file = record_audio(duration=duration)
    audio, sr = librosa.load(output_file, res_type='kaiser_fast')
    real_time_features = extract_features(audio, sr)
    real_time_features = scaler.transform([real_time_features])

    predicted_emotion = rf_best_model.predict(real_time_features)
    predicted_emotion_label = le.inverse_transform(predicted_emotion)

    print(f'Predicted Emotion: {predicted_emotion_label[0]}')

print("Recording and predicting emotion...")
record_and_predict_emotion(duration=20)
