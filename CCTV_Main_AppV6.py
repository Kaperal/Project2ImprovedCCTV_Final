import cv2
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from gun_and_human_detectionV2 import detect_objects
from face_recognition_module import recognize_faces
import threading
from audio_detectV3 import AudioDetection  # Import the audio detection module
import sounddevice as sd

import csv
import datetime
import os

import serial

# Desired display size
d_width = 1080
d_height = 720


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
        self.video_label.grid(row=3, column=0, padx=(100,0) ,columnspan=4, sticky='nsew')

        # Closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Prepare for result saving
        self.prepare_results_folder()

        self.alert_flag = False  # Shared state for detection
        self.gun_detected = False

        # Add frame for simulation buttons
        self.simulation_frame = tk.Frame(root)
        self.simulation_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nw")
        # Simulate Gun button
        self.simulate_gun_button = tk.Button(self.simulation_frame, text="Simulate Gun", command=self.simulate_gun)
        self.simulate_gun_button.pack(pady=5)

        # Simulate Scream button
        self.simulate_scream_button = tk.Button(self.simulation_frame, text="Simulate Scream",
                                                command=self.simulate_scream)
        self.simulate_scream_button.pack(pady=5)
        # Reset Scream and Gun
        self.reset_button = tk.Button(self.simulation_frame, text="Reset Scream and Gun",
                                                command=self.reset_button)
        self.reset_button.pack(pady=5)
        self.TextingTest_button = tk.Button(self.simulation_frame, text="Test Texting",
                                      command=self.test_button)
        self.TextingTest_button.pack(pady=5)


    def show_scream_alert(self):
        # If the alert label already exists, remove it before creating a new one
        if hasattr(self, "scream_label") and self.scream_label:
            self.scream_label.destroy()

        # Create the overlay label
        self.scream_label = tk.Label(self.root, text="SCREAM DETECTED!", fg="white", bg="red",
                                     font=("Arial", 24, "bold"))

        # Place it at the center and in front of everything
        self.scream_label.place(relx=0.95, rely=0.20, anchor="e")

        # Bring it to the front
        self.scream_label.lift()

        # Auto-hide after 3 seconds
        self.scream_label.after(3000, self.hide_scream_alert)

    def show_gun_alert(self):
        # If the alert label already exists, remove it before creating a new one
        if hasattr(self, "gun_label") and self.gun_label:
            self.gun_label.destroy()

        # Create the overlay label
        self.gun_label = tk.Label(self.root, text="WEAPON DETECTED!", fg="white", bg="red",
                                     font=("Arial", 24, "bold"))

        # Place it at the center and in front of everything
        self.gun_label.place(relx=0.95, rely=0.30, anchor="e")

        # Bring it to the front
        self.gun_label.lift()

        # Auto-hide after 3 seconds
        self.gun_label.after(3000, self.hide_scream_alert)

    def hide_scream_alert(self):
        if hasattr(self, "scream_label") and self.scream_label:
            self.scream_label.destroy()
            self.scream_label = None
    def hide_gun_alert(self):
        if hasattr(self, "gun_label") and self.gun_label:
            self.gun_label.destroy()
            self.gun_label = None
    def simulate_gun(self):
        """Simulate a gun detection."""
        self.gun_detected = True
        self.show_gun_alert()
        #self.serial_connection.write(b"Assault;yes;Alwin;7182378:1983920:127829/121/21")
        print("Simulated Gun Detection: gun_detected set to True.")

    def simulate_scream(self):
        """Simulate a scream detection."""
        self.alert_flag = True
        self.show_scream_alert()
        print("Simulated Scream Detection: alert_flag set to True.")
    def reset_button(self):
        """Simulate a gun detection."""
        self.gun_detected = False
        self.alert_flag = False
        self.hide_scream_alert()
        self.hide_gun_alert()

    def test_button(self):
        self.serial_connection.write(b"TEST; yes; TEST; 7182378:1983920:127829/121/21")



    def initialize_audio_model(self):
        """Load an existing trained audio model without retraining."""
        try:
            model_path = "trained_model2.pkl"  # Path to the pre-trained model
            if os.path.exists(model_path):
                print(f"Loading trained audio model from {model_path}...")
                self.audio_detection = AudioDetection(model_path=model_path)  # Load model
                print("Audio detection model loaded successfully.")
            else:
                try:
                    print("Loading and training the audio detection model...")
                    X, y = self.audio_detection.load_data()  # Load dataset
                    self.audio_detection.train_model(X, y)  # Train the model
                    print("Audio detection model is ready.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to initialize audio detection: {e}")
                messagebox.showerror("Error", "Trained audio model not found!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio detection model: {e}")
    def run_audio_detection_thread(self, microphone_id):
        """Run audio detection in a separate thread."""

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            try:
                # Process audio data in the callback
                detected_event = self.audio_detection.detect_event()  # Replace with your detection logic
                if detected_event == "scream":
                    self.alert_flag = True
                    print("Audio event detected: ", detected_event)
                #else:
                    #self.alert_flag = False
            except Exception as e:
                print(f"Error in audio callback: {e}")

        def audio_stream_runner():
            """Thread runner for the audio stream."""
            try:
                with sd.InputStream(device=microphone_id, callback=audio_callback):
                    print("Audio detection started. Listening for events...")
                    sd.sleep(1000000)  # Keep the stream running indefinitely
            except Exception as e:
                print(f"Error in audio detection thread: {e}")

        # Start the audio stream in a new thread
        audio_thread = threading.Thread(target=audio_stream_runner, daemon=True)
        audio_thread.start()

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
        self.selected_port = self.port_dropdown.get()
        if self.selected_port:
            try:
                self.serial_connection = serial.Serial(self.selected_port, baudrate=9600, timeout=1)
                #self.serial_connection.flushInput()
                self.serial_connection.open()

                print(f"Selected port: {self.serial_connection.port}")
                messagebox.showinfo("Success", f"Connected to {self.selected_port}")
            except Exception as e:
                print(f"Error opening serial port: {e}")
                messagebox.showerror("Error", f"Cannot connect to {self.selected_port}: {e}")

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
            try:
                # Process audio data in the callback
                detected_event = self.audio_detection.detect_event()  # Replace with your detection logic
                if detected_event == "scream":
                    self.alert_flag = True
                    self.show_scream_alert()  # Show label
                    print("Audio event detected: ", detected_event)
            except Exception as e:
                print(f"Error in audio callback: {e}")



            ret, frame = self.cap.read()
            if not ret:
                break

            # Step 1: Object detection (guns, humans)
            detections, frame = detect_objects(frame)
            #Step 2: Face Recognition
            faces, frame = recognize_faces(frame)
            # Step 2: Check for guns or humans by iterating through detected objects

            for detection in detections:
                label = detection['label']
                confidence = detection['confidence']
                x1, y1, x2, y2 = detection['bbox']  # Unpack bounding box coordinates



                # Check if a gun is detected
                if label and confidence > 0.5: #Name to be replaced
                    self.gun_detected = True
                    self.show_gun_alert()

                    print("Gun detected! Triggering alert...")

                    if self.serial_connection:
                        self.serial_connection.write(b'ALERT: Gun detected!\n')


                # Draw bounding boxes for detected objects (e.g., gun, human)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                #faces, annotated_frame = recognize_faces(frame)

            for face in faces:
                name = face['name']
                if face['name'] != "Unknown":
                    print(f"Recognized: {face['name']}")
                else:
                    print("Unknown face detected.")

                        # Optionally, add further actions like saving to a log, sending an email, etc.
            # Log the object details to CSV
            # Step 6: Log the detection details to CSV
            # Log the object and face details to CSV
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            label_face = [face['name'] for face in faces]

            labels = [detection['label'] for detection in detections]
            output = f"{', '.join(labels)}; {self.alert_flag}; {', '.join(label_face)}; {current_time} {current_date}"

            with open(self.csv_file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                if self.header == 0:
                    csv_writer.writerow(
                        ["Label", "X coordinate", "Y coordinate", "Confidence", "Time","Faces", "Frame Count","Scream Detected"])
                    self.header = 1

                # Log the object detections
                for detection in detections:
                    label = detection['label']
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    csv_writer.writerow(
                        [label, x1, y1, confidence, current_time, ', '.join(label_face), self.frame_count,self.alert_flag,self.gun_detected])
                if not detections:
                    csv_writer.writerow(["", "", "", "", current_time, ', '.join(label_face), self.frame_count,self.alert_flag,self.gun_detected])
            # Step : Combined Logic
            if self.gun_detected == True and self.alert_flag == True:
                print("ALERT: Gun detected in video AND scream or gunshot detected in audio!")
                self.gun_detected = False
                self.alert_flag = False
                self.hide_scream_alert()
                self.hide_gun_alert()
                self.serial_connection.write(output.encode() + b'\n')

                messagebox.showwarning("It activated")
            # Resize the frame to the desired display size
            frame = cv2.resize(frame, (d_width, d_height))

            # Save the frame to the disk
            frame_path = os.path.join(self.pics_folder_path, f"frame_{self.frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            # Write the frame to the video file
            self.video_writer.write(frame)

            self.frame_count += 1
            # Convert frame to Image for Tkinter display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.root.update_idletasks()
            self.root.update()

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

    def prepare_results_folder(self):
        # Create a folder to save the results if it doesn't exist
        home_dir = os.path.expanduser("~")
        documents_dir = os.path.join(home_dir, "Downloads")
        result_folder_path = os.path.join(documents_dir, 'results_Yolov8')
        os.makedirs(result_folder_path, exist_ok=True)

        # Create a sub-folder for the current run
        timestamp = datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S")
        self.run_folder = os.path.join(result_folder_path, timestamp)
        os.makedirs(self.run_folder, exist_ok=True)

        # Path for CSV and pictures
        self.csv_file_path = os.path.join(self.run_folder, "detection_results.csv")
        self.pics_folder_path = os.path.join(self.run_folder, "frames")
        os.makedirs(self.pics_folder_path, exist_ok=True)

        # Initialize header flag and frame count
        self.header = 0
        self.frame_count = 0

        # Initialize VideoWriter
        video_path = os.path.join(self.run_folder, "output_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (d_width, d_height))

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.serial_inst.is_open:
            self.serial_inst.close()
        if hasattr(self, 'video_writer') and self.video_writer.isOpened():
            self.video_writer.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = CCTVApp(root)
    root.mainloop()

