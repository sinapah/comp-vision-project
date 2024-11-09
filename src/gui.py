#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:48:45 2024

@author: sinap
"""

import os
from tkinter import Label, Button
from PIL import Image, ImageTk

class ImageGallery:
    def __init__(self, master, folder_paths):
        self.master = master
        self.master.title("Image Viewer")
        
        # Load images and labels
        self.images = []
        self.labels = []
        self.load_images(folder_paths)
        
        # Initial index for tracking the current image
        self.current_index = 0

        # Display area for images
        self.image_label = Label(master)
        self.image_label.pack()

        # Display area for arbitrary label
        self.label_display = Label(master, text="", font=("Arial", 16))
        self.label_display.pack()

        # Navigation buttons
        self.prev_button = Button(master, text="<< Previous", command=self.prev_image)
        self.prev_button.pack(side="left")
        
        self.next_button = Button(master, text="Next >>", command=self.next_image)
        self.next_button.pack(side="right")

        # Display the first image
        self.display_image()

    def load_images(self, folder_paths):
        for folder in folder_paths:
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    # Load the image and store it
                    img_path = os.path.join(folder, filename)
                    image = Image.open(img_path)
                    self.images.append(image)

                    # Store an arbitrary label (you can customize these)
                    self.labels.append(f"Image from {folder}: {filename}")
        
    def display_image(self):
        image = self.images[self.current_index]
        image.thumbnail((400, 400))  # Resize to fit in the GUI window

        # Convert image for Tkinter and update display
        img_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Keep reference to avoid garbage collection

        # Update the arbitrary label
        self.label_display.config(text=self.labels[self.current_index])

    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.display_image()
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()



