from PIL import Image
import os
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import warnings
from dateutil.parser import parse

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

    def create_reference_section(self, ref_name, top_left, width, height):
        # Create a section with given parameters
        section = Section(self, top_left, width, height)

        # Create directory to save the reference image if it doesn't exist
        reference_dir = os.path.join(self.base_dir, 'reference_sections')
        os.makedirs(reference_dir, exist_ok=True)

        # Create a PIL Image object from the section's image data
        reference_image = Image.fromarray(section.image_data)

        # Save the reference image
        reference_image_path = os.path.join(reference_dir, f'{ref_name}.png')
        reference_image.save(reference_image_path)

        # Save metadata in a separate JSON file
        metadata = {
            'top_left': top_left,
            'width': width,
            'height': height
        }
        with open(os.path.join(reference_dir, f'{ref_name}_metadata.json'), 'w') as json_file:
            json.dump(metadata, json_file, indent=4)


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

    def create_section_plot(self):
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

    def display_section_plot(self):
        ax = self.create_section_plot()
        plt.tight_layout()
        plt.show()


class ReferenceSection(Section):
    def __init__(self, canvas, filename):
        # Load metadata from JSON file
        metadata_file = os.path.join('canvas_data', canvas.event_name, 'reference_sections',
                                     f'{filename}_metadata.json')
        with open(metadata_file, 'r') as json_file:
            metadata = json.load(json_file)

        # Extract parameters from metadata
        top_left = tuple(metadata['top_left'])
        width = metadata['width']
        height = metadata['height']

        super().__init__(canvas, top_left, width, height)
        self.reference_section_name = filename
        self.reference_image_path = os.path.join('canvas_data', canvas.event_name, 'reference_sections',
                                                 f'{self.reference_section_name}.png')
        self.reference_image_data = self.load_image(self.reference_image_path)

    def load_image(self, img_path):
        img = Image.open(img_path)
        return img.convert("RGB")

    def create_three_part_plot(self, comparison_image_file):
        image_data = self.load_image(comparison_image_file)
        readable_timestamp = comparison_image_file[:-4]
        image_timestamp = readable_timestamp[readable_timestamp.rfind('/') + 1:readable_timestamp.rfind('.')]

        # Create a 1x3 subplot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Convert PIL Images to NumPy arrays
        current_image_data = np.array(image_data)
        reference_image_data = np.array(self.reference_image_data)

        # Display the reference image
        axs[0].imshow(reference_image_data)
        axs[0].set_title("Reference Image")

        # Get coordinates of top left corner of the reference section
        top_left_x = self.top_left[0]
        top_left_y = self.top_left[1]

        # Display the current section of the image
        section_image_data = current_image_data[top_left_y:top_left_y+self.height,
                                                top_left_x:top_left_x+self.width]

        axs[1].imshow(section_image_data)
        axs[1].set_title(image_timestamp)

        # Calculate and display the difference
        diff = np.abs(section_image_data.astype(int) - reference_image_data.astype(int))
        axs[2].imshow(np.any(diff > 0, axis=2), cmap='gray')
        axs[2].set_title("Difference Image")

        # Remove the axis labels for cleaner visualization
        for ax in axs:
            ax.axis('off')

        return fig, axs

    def create_timelapse(self, start_time=None, end_time=None):
        # Create directory for frames if it doesn't exist
        frames_dir = os.path.join('canvas_data', self.canvas.event_name, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Get list of all image files in directory and sort by timestamp
        image_files = sorted(glob.glob(os.path.join(self.canvas.base_dir, 'subcanvas_*', '*.png')),
                             key=lambda x: parse(x.split('/')[-1].split('.png')[0]))

        for image_n, img_file in enumerate(image_files):
            timestamp_str = img_file.split('/')[-1].split('.png')[0]
            timestamp = parse(timestamp_str)

            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue

            fig, axs = self.create_three_part_plot(img_file)
            #fig.suptitle('')

            # Save the figure
            fig.savefig(os.path.join(frames_dir, f'frame{image_n+1}.png'))

            # Close the plot
            plt.close(fig)