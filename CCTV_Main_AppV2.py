import cv2
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import sounddevice as sd
import os
import datetime
import xlsxwriter

class CCTVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CCTV Surveillance System")
        self.root.state('zoomed')  # Maximize window on start

        # Variables for camera and serial port
        self.selected_camera = tk.StringVar()
        self.selected_port = tk.StringVar()
        self.cap = None
        self.serial_connection = None

        # Variables for results and Excel file
        self.results_folder = None
        self.excel_path = None
        self.workbook = None
        self.worksheet = None
        self.excel_row = 1

        # Layout configuration
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        # Dropdown for camera selection
        tk.Label(root, text="Select Camera:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.camera_dropdown = ttk.Combobox(root, textvariable=self.selected_camera, values=self.detect_cameras())
        self.camera_dropdown.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        tk.Button(root, text="Select Camera", command=self.select_camera).grid(row=2, column=0, padx=5, pady=5, sticky='w')

        # Dropdown for serial port selection
        tk.Label(root, text="Select Serial Port:").grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.port_dropdown = ttk.Combobox(root, textvariable=self.selected_port, values=self.detect_serial_ports())
        self.port_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        tk.Button(root, text="Select Port", command=self.select_port).grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # Button to start CCTV
        tk.Button(root, text="Start CCTV", command=self.start_cctv).grid(row=2, column=2, padx=5, pady=5)

        # Video display label
        self.video_label = tk.Label(root)
        self.video_label.grid(row=3, column=0, columnspan=3, sticky='nsew')

        # Closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def detect_cameras(self):
        """Detect available cameras."""
        camera_indices = []
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                camera_indices.append(str(i))
            cap.release()
        return camera_indices if camera_indices else ["No Camera Found"]

    def detect_serial_ports(self):
        """Detect available serial ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ["No Port Found"]

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

    def create_results_folder(self):
        """Create a results folder in Downloads with date and time subfolders."""
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        results_path = os.path.join(downloads_path, "Results")
        os.makedirs(results_path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_folder = os.path.join(results_path, timestamp)
        os.makedirs(session_folder, exist_ok=True)

        print(f"Results folder created at: {session_folder}")
        return session_folder

    def initialize_excel(self, folder_path):
        """Initialize an Excel workbook in the given folder."""
        excel_path = os.path.join(folder_path, "Detection_Results.xlsx")
        self.workbook = xlsxwriter.Workbook(excel_path)
        self.worksheet = self.workbook.add_worksheet("Results")
        self.worksheet.write_row(0, 0, ["Timestamp", "Label", "Image Path"])
        print(f"Excel file created at: {excel_path}")
        return excel_path

    def save_detection_to_excel(self, timestamp, label, image_path):
        """Append detection results to the Excel file."""
        self.worksheet.write_row(self.excel_row, 0, [timestamp, label, image_path])
        self.excel_row += 1

    def start_cctv(self):
        """Start CCTV and initialize folders and files."""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera is not selected or not available.")
            return

        # Initialize results folder and Excel file
        self.results_folder = self.create_results_folder()
        self.excel_path = self.initialize_excel(self.results_folder)

        # Start CCTV monitoring in a separate thread
        cctv_thread = threading.Thread(target=self.run_cctv, daemon=True)
        cctv_thread.start()

    def run_cctv(self):
        """Run the CCTV surveillance loop."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            label = "Detected_Object"  # Replace with actual detection logic
            image_filename = f"{label}_{timestamp}.jpg"
            image_path = os.path.join(self.results_folder, image_filename)

            # Save the current frame with detection
            cv2.imwrite(image_path, frame)
            print(f"Saved image: {image_path}")

            # Save detection data to Excel
            self.save_detection_to_excel(timestamp, label, image_path)

            # Display video feed in GUI
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Cleanup
        if self.cap:
            self.cap.release()
        if self.workbook:
            self.workbook.close()

    def on_close(self):
        """Handle cleanup on window close."""
        if self.cap:
            self.cap.release()
        if self.serial_connection:
            self.serial_connection.close()
        if self.workbook:
            self.workbook.close()
        cv2.destroyAllWindows()
        self.root.quit()

    def prepare_results_folder(self):
        # Create a folder to save the results if it doesn't exist
        home_dir = os.path.expanduser("~")
        documents_dir = os.path.join(home_dir, "Documents")
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
