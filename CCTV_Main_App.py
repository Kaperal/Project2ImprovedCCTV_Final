import cv2
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from gun_and_human_detection import detect_objects
from face_recognition_module import recognize_faces
import threading
from audio_detect import AudioDetection  # Import the audio detection module
import sounddevice as sd

class CCTVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Surveillance System")
        self.root.state('zoomed')  # Maximize window on start

        # Variables for camera, serial port, and microphone selection
        self.selected_camera = tk.StringVar()
        self.selected_port = tk.StringVar()
        self.selected_microphone = tk.StringVar()
        self.cap = None
        self.serial_connection = None

        # Initialize audio detection
        self.audio_detection = AudioDetection()
        self.audio_thread = None
        self.initialize_audio_model()

        # Layout configuration
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        # Dropdown for camera selection
        self.camera_label = tk.Label(root, text="Select Camera:")
        self.camera_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.camera_dropdown = ttk.Combobox(root, textvariable=self.selected_camera)
        self.camera_dropdown['values'] = self.detect_cameras()
        self.camera_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.camera_button = tk.Button(root, text="Select Camera", command=self.select_camera)
        self.camera_button.grid(row=2, column=0, padx=5, pady=5, sticky='w')

        # Dropdown for serial port selection
        self.port_label = tk.Label(root, text="Select Serial Port:")
        self.port_label.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.port_dropdown = ttk.Combobox(root, textvariable=self.selected_port)
        self.port_dropdown['values'] = self.detect_serial_ports()
        self.port_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        self.port_button = tk.Button(root, text="Select Port", command=self.select_port)
        self.port_button.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # Dropdown for microphone selection
        self.microphone_label = tk.Label(root, text="Select Microphone:")
        self.microphone_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

        self.microphone_dropdown = ttk.Combobox(root, textvariable=self.selected_microphone)
        self.microphone_dropdown['values'] = self.detect_microphones()
        self.microphone_dropdown.grid(row=1, column=2, padx=5, pady=5, sticky='w')

        self.microphone_button = tk.Button(root, text="Select Microphone", command=self.select_microphone)
        self.microphone_button.grid(row=2, column=2, padx=5)

        # Button to start CCTV
        self.start_button = tk.Button(root, text="Start CCTV", command=self.start_cctv)
        self.start_button.grid(row=2, column=3, padx=5, pady=5)

        # Video display label
        self.video_label = tk.Label(root)
        self.video_label.grid(row=3, column=0, columnspan=4, sticky='nsew')

        # Closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def initialize_audio_model(self):
        """Load and train the audio detection model."""
        try:
            print("Loading and training the audio detection model...")
            X, y = self.audio_detection.load_data()  # Load data
            self.audio_detection.train_model(X, y)  # Train the model
            print("Audio detection model is ready.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize audio detection: {e}")

    def start_audio_detection(self):
        """Start real-time audio detection in a thread."""
        microphone_id = self.selected_microphone_id  # Ensure microphone is selected
        if microphone_id is None:
            messagebox.showerror("Error", "No microphone selected for audio detection.")
            return

        print("Starting real-time audio detection...")
        self.audio_detection.run_audio_detection_thread(microphone_id)
    def detect_cameras(self):
        """Detects available cameras."""
        camera_indices = []
        for i in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                camera_indices.append(str(i))
            cap.release()
        return camera_indices if camera_indices else ["No Camera Found"]

    def detect_serial_ports(self):
        """Detects available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ["No Port Found"]

    def detect_microphones(self):
        """Detects available microphone devices."""
        devices = sd.query_devices()
        microphones = [device['name'] for device in devices if device['max_input_channels'] > 0]
        return microphones if microphones else ["No Microphone Found"]

    def select_camera(self):
        """Select the camera based on user input."""
        camera_index = self.selected_camera.get()
        if camera_index.isdigit():
            self.cap = cv2.VideoCapture(int(camera_index))
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open selected camera.")
        else:
            messagebox.showwarning("Warning", "Please select a valid camera.")

    def select_port(self):
        """Connect to the selected serial port."""
        port = self.selected_port.get()
        try:
            self.serial_connection = serial.Serial(port, baudrate=9600, timeout=1)
            messagebox.showinfo("Success", f"Connected to {port}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot connect to {port}: {e}")

    def select_microphone(self):
        """Set the selected microphone for audio detection."""
        microphone_name = self.selected_microphone.get()
        devices = sd.query_devices()
        self.selected_microphone_id = None
        for device in devices:
            if device['name'] == microphone_name:
                self.selected_microphone_id = device['index']
                break
        if self.selected_microphone_id is not None:
            messagebox.showinfo("Success", f"Microphone {microphone_name} selected")
        else:
            messagebox.showerror("Error", "Failed to select the microphone")

    def start_cctv(self):
        """Start CCTV and audio detection in separate threads."""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera is not selected or not available.")
            return

        # Start CCTV monitoring
        cctv_thread = threading.Thread(target=self.run_cctv, daemon=True)
        cctv_thread.start()

        # Start audio detection
        if self.selected_microphone_id is not None:
            self.audio_thread = self.audio_detection.run_audio_detection_thread(self.selected_microphone_id)
        else:
            messagebox.showwarning("Warning", "No microphone selected. Audio detection will not run.")

    def run_cctv(self):
        """Run the CCTV surveillance loop."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Step 1: Object detection (guns, humans)
            detections, frame = detect_objects(frame)

            # Step 2: Check for guns or humans by iterating through detected objects
            for detection in detections:
                label = detection['label']

                # Check if a gun is detected
                if label == 'gun':
                    print("Gun detected! Triggering alert...")
                    if self.serial_connection:
                        self.serial_connection.write(b'ALERT: Gun detected!\n')

                # Check if a human is detected
                elif label == 'person':
                    print("Human detected, starting face recognition...")
                    faces, annotated_frame = recognize_faces(frame)
                    for face in faces:
                        if face['name'] != "Unknown":
                            print(f"Recognized: {face['name']}")
                        else:
                            print("Unknown face detected.")
                    break  # Stop processing once a person is detected

            # Convert frame to Image for Tkinter display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.cap:
            self.cap.release()

    def on_close(self):
        """Handle cleanup on window close."""
        if self.cap:
            self.cap.release()
        if self.serial_connection:
            self.serial_connection.close()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)
        cv2.destroyAllWindows()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = CCTVApp(root)
    root.mainloop()
