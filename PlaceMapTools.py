from PIL import Image
import os
import glob

import matplotlib.pyplot as plt
import numpy as np


class Canvas:
    def __init__(self, event_name):
        self.subcanvases = {}
        self.event_name = event_name
        self.base_dir = os.path.join('canvas_data', self.event_name)
        os.makedirs(self.base_dir, exist_ok=True)

    def add_subcanvas(self, id, timestamp=None):
        subcanvas_dir = os.path.join(self.base_dir, f'subcanvas_{id}')
        os.makedirs(subcanvas_dir, exist_ok=True)
        self.subcanvases[id] = SubCanvas(subcanvas_dir, timestamp)

    def generate_canvas_image(self):
        pass
        # Your logic to generate a full canvas image

    def get_diff(self, other_canvas):
        pass
        # Your logic to get diffs


class SubCanvas:
    def __init__(self, directory, timestamp=None):
        self.directory = directory
        if timestamp:
            self.image_path = os.path.join(self.directory, f'image_{timestamp}.png')
        else:
            list_of_files = glob.glob(os.path.join(self.directory, '*.png'))  # Get list of all png files
            self.image_path = max(list_of_files, key=os.path.getctime)  # Get latest file
        self.image_data = self.load_image(self.image_path)

    def load_image(self, img_path):
        img = Image.open(img_path)
        return img.convert("RGB")

    def generate_diff(self, other_subcanvas):
        pass
        # Your logic to generate a diff with another SubCanvas instance

    def display(self):
        # Convert the PIL Image object to a NumPy array
        img_array = np.array(self.image_data)
        print(img_array)
        # Display the image using matplotlib with the right color map
        plt.axis('off')
        plt.imshow(img_array)
        plt.show()
