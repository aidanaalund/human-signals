import os # Processing stuff
import threading
import asyncio
import multiprocessing # Put inference on GPU or separate core
import serial
from pathlib import Path

import numpy as np # ML stuff
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn as sk
from sklearn import metrics

import bleak

import customtkinter # Desktop app stuff
import tkinter as tk
import mediapipe as mp
import cv2 # Kalman filter possibly?
from cv2_enumerate_cameras import enumerate_cameras
import PIL.Image, PIL.ImageTk

from writer_id_torch import WriterRegistry, load_model_bundle
from writer_id_onnx import ONNXWriterRegistry


class WriterIdentityEngine:
    def __init__(
        self,
        run_dir="writer_runs/latest",
        registry_path="writer_runs/registry.json",
        unknown_threshold=0.72,
        device="cpu",
        backend="torch",
        onnx_path="",
        onnx_provider="CPUExecutionProvider",
    ):
        self.run_dir = run_dir
        self.registry_path = registry_path
        self.unknown_threshold = unknown_threshold
        self.device = device
        self.backend = backend
        self.onnx_path = onnx_path
        self.onnx_provider = onnx_provider
        self.registry = None

    def load(self):
        run_dir = Path(self.run_dir)
        if self.backend == "onnx":
            channel_mean = np.load(run_dir / "channel_mean.npy")
            channel_std = np.load(run_dir / "channel_std.npy")
            onnx_path = Path(self.onnx_path) if self.onnx_path else (run_dir / "writer_encoder.onnx")
            self.registry = ONNXWriterRegistry(
                onnx_path=onnx_path,
                channel_mean=channel_mean,
                channel_std=channel_std,
                target_len=96,
                unknown_threshold=self.unknown_threshold,
                providers=[self.onnx_provider],
            )
        else:
            model, channel_mean, channel_std, _ = load_model_bundle(run_dir, device=self.device)
            self.registry = WriterRegistry(
                model=model,
                channel_mean=channel_mean,
                channel_std=channel_std,
                target_len=96,
                unknown_threshold=self.unknown_threshold,
                device=self.device,
            )

        if os.path.exists(self.registry_path):
            self.registry.load_registry(Path(self.registry_path))

    def save(self):
        if self.registry is not None:
            self.registry.save_registry(Path(self.registry_path))

    def predict_segment(self, segment):
        if self.registry is None:
            self.load()
        writer_id, score = self.registry.predict_or_unknown(segment)
        return writer_id, score

    def enroll_segment(self, writer_id, segment):
        if self.registry is None:
            self.load()
        self.registry.update_writer(writer_id=writer_id, segment=segment)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # TODO: quick fix. search using HW API later
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(f"{self.width}x{self.height}")

    def get_frame(self):
        if not self.vid.isOpened():
            return (False, None)

        return_value, frame = self.vid.read()
        if return_value:
            # Return a boolean success flag and the current frame converted to BGR
            return (return_value, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            return (return_value, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class MyDigitalWhiteboard:
    def __init__(self):
        # Setup the canvas
        self.width = 1280
        self.height = 720

    def add_drawings(self):
        return # TODO
    
    def add_participant(self):
        return # TODO:

    # Release the video source when the object is destroyed
    def __del__(self):
        return # TODO:

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        backend = os.getenv("WRITER_BACKEND", "torch").strip().lower()
        onnx_path = os.getenv("WRITER_ONNX_PATH", "")
        onnx_provider = os.getenv("WRITER_ONNX_PROVIDER", "CPUExecutionProvider") # TODO: change to GPU when possible

        self.writer_engine = WriterIdentityEngine(
            backend=backend,
            onnx_path=onnx_path,
            onnx_provider=onnx_provider,
        )
        self.pending_enroll_writer = None

        # Configure window
        self.title("Whiteboard Digitizer")

        # Make the window fullscreen
        self.after(0, lambda: self.wm_state('zoomed'))

        # Configure grid layout (3 columns, 1 main row or multiple rows)
        self.grid_columnconfigure(0, weight=0) # Settings (fixed width)
        self.grid_columnconfigure(1, weight=1) # Webcam (expandable)
        self.grid_columnconfigure(2, weight=1) # Whiteboard (expandable)
        self.grid_rowconfigure((0, 1), weight=1) # Configure rows to have equal weight

        # --- Sidebar Frame ---
        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(1, weight=1) # Make output section expand

        # --- Controls Section ---
        self.controls_frame = customtkinter.CTkFrame(self.sidebar_frame)
        self.controls_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        self.controls_label = customtkinter.CTkLabel(self.controls_frame, text="Controls", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.controls_label.pack(pady=(10, 20))

        #get the available video devices
        self.camera_list = enumerate_cameras()

        # fill combobox with availible video devices
        self.combobox = customtkinter.CTkOptionMenu(self.controls_frame, values=[c.name for c in self.camera_list])
        self.combobox.pack(pady=10, padx=20, fill="x")

        # Recording button
        self.camera_button = customtkinter.CTkButton(self.controls_frame, text="Connect camera", command=self.sidebar_button_event)
        self.camera_button.pack(pady=10, padx=20, fill="x")

        # Pen button
        self.pen_button = customtkinter.CTkButton(self.controls_frame, text="Connect to pen", command=self.bluetooth_callback)
        self.pen_button.pack(pady=10, padx=20, fill="x")

        # Character Recognition button
        self.inference_button = customtkinter.CTkButton(self.controls_frame, text="Begin inference", command=None) # TODO
        self.inference_button.pack(pady=10, padx=20, fill="x")

        # --- Output Section ---
        self.output_frame = customtkinter.CTkFrame(self.sidebar_frame)
        self.output_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        self.output_label = customtkinter.CTkLabel(self.output_frame, text="Output", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.output_label.pack(pady=(10, 20))

        self.enroll_name_entry = customtkinter.CTkEntry(self.output_frame, placeholder_text="New/Existing User ID")
        self.enroll_name_entry.pack(pady=10, padx=20, fill="x")

        self.enroll_btn = customtkinter.CTkButton(self.output_frame, text="Enroll next stroke", command=self.arm_enrollment)
        self.enroll_btn.pack(pady=10, padx=20, fill="x")

        self.id_label = customtkinter.CTkLabel(self.output_frame, text="Writer: unknown")
        self.id_label.pack(pady=10, padx=20)

        # --- Settings Section ---
        self.settings_frame = customtkinter.CTkFrame(self.sidebar_frame)
        self.settings_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.settings_label = customtkinter.CTkLabel(self.settings_frame, text="Settings", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.settings_label.pack(pady=(10, 20))

        self.checkbox = customtkinter.CTkCheckBox(self.settings_frame, text="GPU Acceleration")
        self.checkbox.pack(pady=10, padx=20)

        # --- Main Content ---
        self.whiteboard_canvas = tk.Canvas(self, width=640, height=480, bg="white")
        self.whiteboard_canvas.grid(row=0, column=2, rowspan=2, padx=20, pady=20, sticky="nsew")
        # self.whiteboard = MyDigitalWhiteboard() # TODO:

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self, width=640, height=480, bg="white") # NOTE: fixed webcam resolution...
        self.canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.snapshot = tk.Canvas(self, width=640, height=480, bg="grey") # NOTE: fixed webcam resolution...
        self.snapshot.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
    
    def sidebar_button_event(self):
        print("try to open camera: " + self.combobox.get())   

        selected_name = self.combobox.get()
        self.video_source = 0  # default fallback
        for i, device in enumerate(self.camera_list):   
            if device.name == selected_name:
                self.video_source = i
                break

        # main window
        self.vid = MyVideoCapture(self.video_source)
        self.recording = True

        self.delay = 15
        self.update_camera() 

    def update_camera(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        interval = False # TODO: decide how often we take pictures

        if ret:
            # Resize frame to fit the canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1: # Avoid resizing to 1x1
                frame = cv2.resize(frame, (canvas_width, canvas_height))

            if interval:
                self.process_frame()
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            
            # Clear previous image
            self.canvas.delete("all")
            # Center the image on the canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.after(self.delay, self.update_camera)

    def bluetooth_callback(self):
        print("Connecting to pen...")
        self.writer_engine.load()

    def arm_enrollment(self):
        writer_id = self.enroll_name_entry.get().strip()
        if not writer_id:
            self.id_label.configure(text="Writer: enter user id first")
            return
        self.pending_enroll_writer = writer_id
        self.id_label.configure(text=f"Writer: waiting to enroll {writer_id}")

    def on_imu_segment_ready(self, segment):
        """
        Hook this into your serial IMU segmentation callback.
        segment shape should be (T, 6).
        """
        if self.pending_enroll_writer is not None:
            writer_id = self.pending_enroll_writer
            self.writer_engine.enroll_segment(writer_id, segment)
            self.writer_engine.save()
            self.pending_enroll_writer = None
            self.id_label.configure(text=f"Writer: enrolled {writer_id}")
            return

        writer_id, score = self.writer_engine.predict_segment(segment)
        if writer_id is None:
            self.id_label.configure(text=f"Writer: unknown ({score:.2f})")
            return

        self.writer_engine.enroll_segment(writer_id, segment)
        self.id_label.configure(text=f"Writer: {writer_id} ({score:.2f})")

    def process_frame(self):
        return # TODO

if __name__ == "__main__":
    app = App()
    app.mainloop()