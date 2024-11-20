import os
import sounddevice as sd
import wave
from tkinter import *
from tkinter import ttk, messagebox
from datetime import datetime


class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder Application")
        self.root.geometry("500x300")

        self.selected_audio_device = None
        self.final_folder_path = None  # Final folder path for saving audio files

        # Create "Audio_Files_data" folder in the current directory if it doesn't exist
        self.base_folder = os.path.join(os.getcwd(), "Audio_Files_data")
        os.makedirs(self.base_folder, exist_ok=True)

        # Main frame for controls
        controls_frame = Frame(self.root)
        controls_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

        # Audio source selection
        Label(controls_frame, text="Select Audio Source:").pack(anchor=W, pady=5)
        self.audio_dropdown = ttk.Combobox(controls_frame, state='readonly')
        self.audio_dropdown.pack(fill=X, pady=5)
        self.audio_dropdown['values'] = self.get_available_audio_sources()
        self.audio_dropdown.current(0)

        self.audio_button = Button(controls_frame, text="Select Audio Source", command=self.select_audio_source)
        self.audio_button.pack(fill=X, pady=10)

        # Dropdown for selecting audio category
        Label(controls_frame, text="Select Audio Category:").pack(anchor=W, pady=5)
        self.category_dropdown = ttk.Combobox(controls_frame, state='readonly')
        self.category_dropdown.pack(fill=X, pady=5)
        self.category_dropdown['values'] = ["background", "gunshot", "scream"]
        self.category_dropdown.current(0)

        # Record button
        self.record_button = Button(controls_frame, text="Record Audio", command=self.record_audio)
        self.record_button.pack(fill=X, pady=20)

    def get_available_audio_sources(self):
        devices = sd.query_devices()
        input_devices = [f"{idx}: {device['name']}" for idx, device in enumerate(devices) if device['max_input_channels'] > 0]
        return input_devices

    def select_audio_source(self):
        selected_device = self.audio_dropdown.get()
        if not selected_device:
            messagebox.showwarning("Warning", "Please select an audio source!")
            return
        self.selected_audio_device = int(selected_device.split(":")[0])
        messagebox.showinfo("Info", f"Selected audio source: {selected_device.split(':', 1)[1]}")

    def record_audio(self):
        if self.selected_audio_device is None:
            messagebox.showwarning("Warning", "Please select an audio source first!")
            return

        # Get selected category from the dropdown
        selected_category = self.category_dropdown.get().strip()
        if not selected_category:
            messagebox.showwarning("Warning", "Please select a valid audio category!")
            return

        # Define the final folder path inside "Audio_Files_data"
        self.final_folder_path = os.path.join(self.base_folder, selected_category)
        os.makedirs(self.final_folder_path, exist_ok=True)

        # Define audio file path
        audio_filename = f"recorded_audio_{datetime.now().strftime('%H-%M-%S')}.wav"
        file_path = os.path.join(self.final_folder_path, audio_filename)

        # Record audio for 10 seconds
        duration = 10  # seconds
        fs = 44100  # Sampling frequency
        try:
            messagebox.showinfo("Recording", "Recording will start now. Please speak!")
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16', device=self.selected_audio_device)
            sd.wait()  # Wait for recording to finish
            self.save_audio_file(file_path, audio_data, fs)
            messagebox.showinfo("Success", f"Audio recorded and saved at {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to record audio: {e}")

    def save_audio_file(self, file_path, audio_data, sample_rate):
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

    def on_closing(self):
        self.root.destroy()


# Initialize and run the application
root = Tk()
app = AudioRecorderApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
