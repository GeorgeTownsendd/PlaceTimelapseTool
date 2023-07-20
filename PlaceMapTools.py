from PIL import Image
import os
import glob
import time
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
import warnings

class Canvas:
    def __init__(self, event_name, datetime_arg=None):
        self.subcanvases = {}
        self.event_name = event_name
        self.base_dir = os.path.join('canvas_data', self.event_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self.datetime_arg = datetime_arg  # datetime argument for SubCanvas

        # Automatically load existing subcanvases
        subcanvas_dirs = glob.glob(os.path.join(self.base_dir, 'subcanvas_*'))
        for dir in subcanvas_dirs:
            id = os.path.basename(dir).replace('subcanvas_', '')  # Extract id from directory name
            self.add_subcanvas(id, self.datetime_arg)


    def add_subcanvas(self, id, timestamp=None):
        subcanvas_dir = os.path.join(self.base_dir, f'subcanvas_{id}')
        os.makedirs(subcanvas_dir, exist_ok=True)
        self.subcanvases[id] = SubCanvas(subcanvas_dir, self.datetime_arg)

    def generate_canvas_image(self):
        pass
        # Your logic to generate a full canvas image

    def get_diff(self, other_canvas):
        pass
        # Your logic to get diffs


class SubCanvas:
    def __init__(self, directory, datetime_arg=None):
        self.directory = directory
        if datetime_arg:
            self.image_path = self.find_closest_image(datetime_arg)
            if not self.image_path:
                warnings.warn(f"No image near specified datetime {datetime_arg} found in {self.directory}")
        else:
            list_of_files = sorted(glob.glob(os.path.join(self.directory, '*.png')), key=os.path.getctime)  # Get list of all png files and sort by creation time
            self.image_path = list_of_files[-1] if list_of_files else None  # Get latest file
        self.image_data = self.load_image(self.image_path) if self.image_path else None

    def find_closest_image(self, target_datetime):
        list_of_files = glob.glob(os.path.join(self.directory, '*.png'))  # Get list of all png files
        if not list_of_files:
            return None
        # Extract timestamps from filenames and convert to datetime objects
        datetimes = [datetime.strptime(os.path.splitext(os.path.basename(file))[0], '%Y-%m-%d %H:%M:%S.%f') for file in list_of_files]
        # Find index of closest datetime to target_datetime
        closest_index = min(range(len(datetimes)), key=lambda i: abs(datetimes[i] - target_datetime))
        return list_of_files[closest_index]

    def load_image(self, img_path):
        img = Image.open(img_path)
        return img.convert("RGB")

    def previous_version(self, index):
        list_of_files = sorted(glob.glob(os.path.join(self.directory, '*.png')), key=os.path.getctime)  # Get list of all png files and sort by creation time
        if index < 0:
            index = len(list_of_files) + index
        image_path = list_of_files[index] if 0 <= index < len(list_of_files) else None
        timestamp = os.path.basename(image_path).replace('image_', '').replace('.png', '') if image_path else None
        return SubCanvas(self.directory, timestamp)

    def generate_diff(self, other_subcanvas):
        img1_array = np.array(self.image_data)
        img2_array = np.array(other_subcanvas.image_data)
        assert img1_array.shape == img2_array.shape, "Images are not the same size"
        diff = np.abs(img1_array - img2_array)
        diff_bool = np.any(diff > 0, axis=2)
        return diff_bool

    def display(self):
        img_array = np.array(self.image_data)
        plt.axis('off')
        plt.imshow(img_array)
        plt.show()


class Section:
    def __init__(self, canvas, top_left, width, height):
        self.canvas = canvas
        self.top_left = top_left
        self.width = width
        self.height = height
        self.subcanvases = self._get_subcanvases()
        self.update_time = None
        self.image_data = np.zeros((height, width, 3), dtype=np.uint8)
        self._fetch_image_data()

    def _get_subcanvases(self):
        subcanvas_ids = []
        x_min = self.top_left[0] // 1000
        x_max = (self.top_left[0] + self.width - 1) // 1000
        y_min = self.top_left[1] // 1000
        y_max = (self.top_left[1] + self.height - 1) // 1000
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                id = f"{i}{j}"  # format indices to match the '00' format
                subcanvas_ids.append(id)
        return subcanvas_ids

    def _fetch_image_data(self):
        for subcanvas_id in self.subcanvases:
            subcanvas = self.canvas.subcanvases.get(subcanvas_id)
            if subcanvas is None:
                continue
            # Calculate the coordinates within the section's image_data array where the subcanvas's image data will be copied
            x_offset = int(subcanvas_id[0]) * 1000 - self.top_left[0]
            y_offset = int(subcanvas_id[1]) * 1000 - self.top_left[1]
            x_slice = slice(max(0, x_offset), min(self.width, x_offset + 1000))
            y_slice = slice(max(0, y_offset), min(self.height, y_offset + 1000))

            # Calculate the coordinates within the subcanvas's image data that will be copied
            subcanvas_x_slice = slice(max(0, -x_offset), min(1000, self.width - x_offset))
            subcanvas_y_slice = slice(max(0, -y_offset), min(1000, self.height - y_offset))

            # Copy the image data from the subcanvas to the section
            self.image_data[y_slice, x_slice] = np.array(subcanvas.image_data)[subcanvas_y_slice, subcanvas_x_slice]

        self.update_time = time.time()

    def update_state(self):
        self._fetch_image_data()

    from matplotlib.ticker import MaxNLocator

    def create_plot(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image_data)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # Get the automatically set ticks
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()

        # Shift the ticks by the top left position to convert them into world coordinates
        x_labels = [int(self.top_left[0] + i) for i in x_ticks]
        y_labels = [int(self.top_left[1] + i) for i in y_ticks]

        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticklabels(y_labels)

        return ax

    def display(self):
        ax = self.create_plot()
        plt.show()


class ReferenceSection(Section):
    def __init__(self, canvas, top_left, width, height, reference_section_name):
        super().__init__(canvas, top_left, width, height)
        self.reference_section_name = reference_section_name
        self.reference_image_path = os.path.join('canvas_data', canvas.event_name, 'reference_sections', f'{self.reference_section_name}.png')
        self.reference_image_data = self.load_image(self.reference_image_path)

    def load_image(self, img_path):
        img = Image.open(img_path)
        return img.convert("RGB")
