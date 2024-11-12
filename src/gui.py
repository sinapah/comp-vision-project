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
        
        # Bind the resize event to dynamically resize the image
        self.master.bind("<Configure>", self.resize_image)
        
        # Load images and labels
        self.images = []
        self.labels = []
        self.load_images(folder_paths)
        
        # Initial index for tracking the current image
        self.current_index = 0

        # Display area for images
        self.image_label = Label(master)
        self.image_label.pack(expand=True, fill="both", pady=(20, 0))
        
        # Label to show the current image number
        self.image_number_label = Label(master, text="", font=("Arial", 12))
        self.image_number_label.pack()
        
        # Display area for arbitrary label
        self.label_display = Label(master, text="", font=("Arial", 16))
        self.label_display.pack()

        # Navigation buttons
        self.prev_button = Button(master, text="<< Previous", command=self.prev_image)
        self.prev_button.pack(side="left", pady=(0, 10))
        
        self.next_button = Button(master, text="Next >>", command=self.next_image)
        self.next_button.pack(side="right", pady=(0, 10))
        
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
                    self.labels.append("Label goes here")
        
    def display_image(self):
        # Update the label with the current image resized to fit the window
        self.update_resized_image()

        # Update the arbitrary label
        self.label_display.config(text=self.labels[self.current_index])
        
        # Update the image number label
        self.image_number_label.config(text=f"Image {self.current_index + 1} of {len(self.images)}")

    def update_resized_image(self):
        # Calculate 50% of the current window size
        width = self.master.winfo_width() // 2
        height = self.master.winfo_height() // 2

        # Resize the current image to take up 50% of the window's width and height
        image = self.images[self.current_index]
        resized_image = image.copy()
        resized_image = resized_image.resize((width, height), Image.ANTIALIAS)

        # Convert the resized image to a format compatible with Tkinter
        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image  # Keep a reference to avoid garbage collection

    def resize_image(self, event):
        # Called whenever the window is resized
        self.update_resized_image()

    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.display_image()
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()



