from PIL import Image
import os
import glob


class Canvas:
    def __init__(self, event_name):
        self.subcanvases = {}
        self.event_name = event_name
        self.base_dir = os.path.join('canvas_data', self.event_name)
        os.makedirs(self.base_dir, exist_ok=True)

    def add_subcanvas(self, coordinates, timestamp=None):
        quadrant_dir = os.path.join(self.base_dir, f'quadrant_{coordinates[0]}_{coordinates[1]}')
        os.makedirs(quadrant_dir, exist_ok=True)
        self.subcanvases[coordinates] = SubCanvas(quadrant_dir, timestamp)

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
        return Image.open(img_path)

    def generate_diff(self, other_subcanvas):
        pass
        # Your logic to generate a diff with another SubCanvas instance

