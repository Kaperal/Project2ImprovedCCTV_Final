import librosa
import numpy as np
import os
from sklearn.svm import SVC  # Using SVM instead of RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving and loading the model
import sounddevice as sd
import threading
import time
import tkinter as tk
from tkinter import ttk

class AudioDetection:
    def __init__(self, model_path=None):
        """Initialize AudioDetection object."""
        self.model_path = model_path
        self.predicted=""
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = SVC(kernel='linear')  # Using SVM with a linear kernel

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
        background_features, background_labels = self.load_files_from_directory(base_folder + 'background', 'background')

        # Combine the features and labels
        X = scream_features + background_features
        y = scream_labels + background_labels

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        return X, y

    def train_model(self, X, y, save_path="trained_model2.pkl"):
        """Train an SVM model with the given data and save the model."""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model using SVM
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
                self.predicted = self.model.predict(mfcc_mean)
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
    def detect_event(self):
        output = self.predicted[0]
        return output


class AudioDetectionGUI:
    def __init__(self, root, audio_detection):
        self.root = root
        self.audio_detection = audio_detection
        self.root.title("Audio Detection System")

        # Dropdown for selecting the microphone device
        self.device_label = tk.Label(root, text="Select Audio Capture Device:")
        self.device_label.pack(padx=10, pady=5)

        self.device_dropdown = ttk.Combobox(root, state="readonly")
        self.device_dropdown.pack(padx=10, pady=5)

        # Populate dropdown with available sound devices
        self.devices = self.get_audio_devices()
        self.device_dropdown['values'] = self.devices
        self.device_dropdown.current(0)  # Set the default device

        # Start detection button
        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(padx=10, pady=10)

    def get_audio_devices(self):
        """Get the list of available audio devices."""
        devices = sd.query_devices()
        device_names = [device['name'] for device in devices if device['max_input_channels'] > 0]
        return device_names

    def start_detection(self):
        """Start the detection process."""
        selected_device_name = self.device_dropdown.get()
        selected_device_id = self.get_device_id_by_name(selected_device_name)

        # Start the detection in a separate thread
        self.audio_detection.run_audio_detection_thread(microphone_id=selected_device_id)

    def get_device_id_by_name(self, device_name):
        """Get the device ID based on the device name."""
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['name'] == device_name:
                return idx
        return None


if __name__ == "__main__":
    # Define the path to save/load the model
    model_path = "trained_model2.pkl"

    # Initialize the audio detection system
    audio_detection = AudioDetection(model_path=model_path)

    # Load data and train model if it doesn't already exist
    if not os.path.exists(model_path):
        X, y = audio_detection.load_data()
        audio_detection.train_model(X, y, save_path=model_path)

    # Create the main window for the GUI
    root = tk.Tk()
    gui = AudioDetectionGUI(root, audio_detection)

    # Start the GUI event loop
    root.mainloop()
