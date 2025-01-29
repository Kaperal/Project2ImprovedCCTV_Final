import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving and loading the model
import sounddevice as sd
import threading
import time

class AudioDetection:
    def __init__(self, model_path=None):
        """Initialize AudioDetection object."""
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = RandomForestClassifier()

    def extract_features(self, file_path):
        """Extract MFCC features from an audio file."""
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean

    def load_files_from_directory(self, directory, label):
        """Load audio files from a directory and extract features."""
        features_list = []
        labels_list = []
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                file_path = os.path.join(directory, filename)
                features = self.extract_features(file_path)
                features_list.append(features)
                labels_list.append(label)
        return features_list, labels_list

    def load_data(self, base_folder="Audio_Files_data/"):
        """Load training data from subfolders within the base folder."""

        scream_features, scream_labels = self.load_files_from_directory(base_folder + 'scream', 'scream')
        gunshot_features, gunshot_labels = self.load_files_from_directory(base_folder + 'gunshot', 'gunshot')
        background_features, background_labels = self.load_files_from_directory(base_folder + 'background', 'background')

        # Combine the features and labels
        X = scream_features + gunshot_features + background_features
        y = scream_labels + gunshot_labels + background_labels

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        return X, y

    def train_model(self, X, y, save_path="trained_model.pkl"):
        """Train a RandomForestClassifier with the given data and save the model."""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Test the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Audio Classifier Accuracy: {accuracy:.2f}")

        # Save the trained model
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")

        return self.model

    def start_audio_detection(self, microphone_id=None):
        """Start real-time audio detection."""
        while True:
            try:
                print("Recording...")
                duration = 2  # seconds
                sr = 22050
                audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, device=microphone_id)
                sd.wait()
                print("Recording finished.")
                audio = audio.flatten()
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
                prediction = self.model.predict(mfcc_mean)
                print(f"Detected Sound: {prediction[0]}")

                # Add a small delay to prevent overloading the CPU
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in audio detection: {e}")
                break

    def run_audio_detection_thread(self, microphone_id=None):
        """Run the audio detection in a separate thread."""
        audio_thread = threading.Thread(target=self.start_audio_detection, args=(microphone_id,), daemon=True)
        audio_thread.start()
        return audio_thread

if __name__ == "__main__":
    # Define the path to save/load the model
    model_path = "trained_model.pkl"

    # Initialize the audio detection system
    audio_detection = AudioDetection(model_path=model_path)

    # Load data and train model if it doesn't already exist
    if not os.path.exists(model_path):
        X, y = audio_detection.load_data()
        audio_detection.train_model(X, y, save_path=model_path)

    # Start real-time audio detection
    audio_detection.run_audio_detection_thread()

    # Keep the main thread alive to continue audio detection
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping audio detection.")
