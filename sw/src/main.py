import os # Processing stuff
import threading
import asyncio
import multiprocessing # Put inference on GPU or separate core
import serial

import numpy as np # ML stuff
import keras
from keras import layers
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn as sk
from sklearn import metrics
import cv2 # Kalman filter possibly?

from pygrabber.dshow_graph import FilterGraph # Webcam stuff
import PIL.Image, PIL.ImageTk

import customtkinter # Desktop app stuff
import tkinter as tk

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(f"{self.width}x{self.height}")

    def get_frame(self):
        if not self.vid.isOpened():
            return (return_value, None)

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

        # Configure window
        self.title("Sketch Board App")
        self.geometry("1920x1080")

        # Configure grid layout (3 columns, 1 main row or multiple rows)
        self.grid_columnconfigure(0, weight=0) # Settings (fixed width)
        self.grid_columnconfigure(1, weight=1) # Webcam (expandable)
        self.grid_columnconfigure(2, weight=1) # Whiteboard (expandable)
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Settings", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        #get the available video devices
        self.graph = FilterGraph()

        # fill combobox with availible video devices
        self.combobox = customtkinter.CTkOptionMenu(self.sidebar_frame, values=self.graph.get_input_devices())
        self.combobox.grid(row=1, column=0, padx=20, pady=(20, 10))

        # Recording button
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Connect camera", command=self.sidebar_button_event)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10)

        # Character Recognition output
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Begin inference", command=None) # TODO
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=(10,0))
        # self.qr_label = customtkinter.CTkLabel(self.sidebar_frame, text="Recognizing...")
        # self.qr_label.grid(row=4, column=0, padx=20, pady=(10, 0))

        # Settings buttons
        self.button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Connect to pen", command=self.bluetooth_callback)
        self.button_2.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="sw")
        
        self.checkbox = customtkinter.CTkCheckBox(self.sidebar_frame, text="GPU Acceleration")
        self.checkbox.grid(row=6, column=0, padx=20, pady=(10, 20), sticky="sw")

        self.whiteboard_canvas = tk.Canvas(self, width=640, height=480, bg="white")
        self.whiteboard_canvas.grid(row=0, column=2, rowspan=5, padx=20, pady=20, sticky="nsew")
        # self.whiteboard = MyDigitalWhiteboard()

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self, width=640, height=480, bg="black") # NOTE: fixed webcam resolution...
        self.canvas.grid(row=0, column=1, rowspan=5, padx=20, pady=20, sticky="nsew")
    
    def sidebar_button_event(self):
        print("try to open camera: " + self.combobox.get())   

        for i, device in enumerate(self.graph.get_input_devices() ):   
            if device == self.combobox.get():
                self.video_source = i

        # main window
        self.vid = MyVideoCapture(self.video_source)

        self.delay = 15
        self.update_camera() 

    def update_camera(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        interval = False # TODO: decide how often we take pictures

        if ret:
            # Resize or process frame here if necessary
            if interval:
                self.process_frame()
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            
            # Center the image on the canvas dynamically
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor=tk.CENTER)

        self.after(self.delay, self.update_camera)

    def bluetooth_callback(self):
        print("Connecting to pen...")

    def process_frame(self):
        return # TODO

if __name__ == "__main__":
    app = App()
    app.mainloop()