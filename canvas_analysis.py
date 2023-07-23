import os
import json
import shutil
import glob
from typing import List
from PIL import Image
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import subprocess
import uuid
import requests
import io


class Coordinate:
    def __init__(self, x: int, y: int, coord_type: str):
        """Initialize a Coordinate object."""
        if coord_type == "image":
            self.image_coord = (x, y)
            self.reddit_coord = self.image_to_reddit_coordinates((x, y))
            self.template_coord = self.image_to_template_coordinates((x, y))
        elif coord_type == "reddit":
            self.reddit_coord = (x, y)
            self.image_coord = self.reddit_to_image_coordinates((x, y))
            self.template_coord = self.image_to_template_coordinates(self.image_coord)
        elif coord_type == "template":
            self.template_coord = (x, y)
            self.image_coord = self.template_to_image_coordinates((x, y))
            self.reddit_coord = self.image_to_reddit_coordinates(self.image_coord)
        else:
            raise ValueError("Invalid coordinate type. Choose from 'image', 'reddit', or 'template'.")

    @staticmethod
    def reddit_to_image_coordinates(coord: tuple) -> tuple:
        return coord[0] + 1500, coord[1] + 1000

    @staticmethod
    def image_to_reddit_coordinates(coord: tuple) -> tuple:
        return coord[0] - 1500, coord[1] - 1000

    @staticmethod
    def template_to_image_coordinates(coord: tuple) -> tuple:
        return coord[0] + 500, coord[1]

    @staticmethod
    def image_to_template_coordinates(coord: tuple) -> tuple:
        return coord[0] - 1000, coord[1] - 500

    @staticmethod
    def reddit_to_template_coordinates(coord: tuple) -> tuple:
        image_coord = Coordinate.reddit_to_image_coordinates(*coord)
        template_coord = Coordinate.image_to_template_coordinates(*image_coord)
        return template_coord

    @staticmethod
    def template_to_reddit_coordinates(coord: tuple) -> tuple:
        image_coord = Coordinate.template_to_image_coordinates(*coord)
        reddit_coord = Coordinate.image_to_reddit_coordinates(*coord)
        return reddit_coord

    def get_image_coordinates(self):
        return self.image_coord

    def get_reddit_coordinates(self):
        return self.reddit_coord

    def get_template_coordinates(self):
        return self.template_coord


class Canvas:
    """Represents a snapshot of the canvas at a particular time."""
    def __init__(self, filename: str, event, load_image: bool = False):
        self.filename = filename
        self.event = event
        self.image = None
        self.timestamp = self._extract_timestamp_from_filename()

        if load_image:
            self.load_image()

    def _extract_timestamp_from_filename(self):
        timestamp_str = os.path.basename(self.filename).split('.')[0]
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

    def load_image(self):
        if isinstance(self.image, type(None)):
            self.image = Image.open(self.filename)

            if self.image.mode != 'RGBA':
                self.image = self.image.convert('RGBA')

        if np.all(self.image == np.array([255, 255, 255, 255])):
            print(f"The canvas instance at {self.timestamp} may be missing")

    def add_reference_section(self, section_name: str, top_left: Coordinate, width: int, height: int):
        reference_section = ReferenceSection.from_canvas(self.event.event_name, section_name, top_left, width, height, self)
        section_dir = os.path.join(self.event.reference_section_dir, section_name)
        print(section_dir)
        os.makedirs(section_dir, exist_ok=True)
        reference_section.save()
        self.event.reference_sections[section_name] = reference_section


class Section:
    """Represents a defined section on the canvas."""
    def __init__(self, top_left: Coordinate, width: int, height: int):
        self.top_left = top_left
        self.width = width
        self.height = height
        self.bottom_right = (self.top_left.get_image_coordinates()[0] + width, self.top_left.get_image_coordinates()[1] + height)

    def extract_from_canvas(self, canvas: Canvas):
        canvas.load_image()

        extracted_section = canvas.image.crop((*self.top_left.get_image_coordinates(), *self.bottom_right))

        if np.all(extracted_section == np.array([255, 255, 255, 255])):
            print(f"The reference section {self.top_left} may be missing")


        return extracted_section

class ReferenceSection(Section):
    """Represents a reference section on the canvas, containing the 'correct' state of the section."""
    def __init__(self, event_name: str, name: str, top_left: Coordinate, width: int, height: int, correct_image: Image.Image, contact="Tim Army (Hello Internet)"):
        super().__init__(top_left, width, height)
        self.correct_image = correct_image
        self.contact = contact
        self.name = name
        self.event_name = event_name
        self.path = os.path.join('canvas_data', self.event_name, 'reference_sections', self.name)

    def get_correct_image_as_numpy(self) -> np.ndarray:
        image = self.correct_image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        return np.array(image)

    @classmethod
    def from_canvas(cls, event_name: str, name: str, top_left: Coordinate, width: int, height: int, canvas: Canvas):
        canvas.load_image()
        bottom_right = Coordinate(top_left.get_image_coordinates()[0] + width, top_left.get_image_coordinates()[1] + height, 'image')
        section = cls(event_name, name, top_left, width, height, canvas.image.crop((*top_left.get_image_coordinates(), *bottom_right.get_image_coordinates())))

        return section

    def save(self):
        with open(os.path.join(self.path, f"{self.name}.json"), 'w') as f:
            json.dump({
                'contact': 'Tim Army (Hello Internet)',
                'templates': [{
                    'name': self.name,
                    'sources': [os.path.join(self.path, f"{self.name}.png")],
                    'x': self.top_left.get_template_coordinates()[0],
                    'y': self.top_left.get_template_coordinates()[1],
                    'top_left_image': self.top_left.get_image_coordinates(),
                    'top_left_reddit': self.top_left.get_reddit_coordinates(),
                    'top_left_template': self.top_left.get_template_coordinates(),
                    'width': self.width,
                    'height': self.height
                }],
                'whitelist': [],
                'blacklist': []
            }, f)
        # Ensure directory exists before saving image
        os.makedirs(self.path, exist_ok=True)
        # Save the image in PNG format
        self.correct_image.save(os.path.join(self.path, f"{self.name}.png"))

    @classmethod
    def load(cls, path: str) -> 'ReferenceSection':
        event_name = path.split('/')[1]
        section_name = os.path.basename(path)

        image_path = os.path.join(path, f"{section_name}.png")
        json_path = os.path.join(path, f"{section_name}.json")
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            print(f"Warning: Skipping loading of section {section_name} due to missing files.")
            return None

        correct_image = Image.open(image_path)
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        template_metadata = metadata['templates'][0]
        top_left_image = template_metadata.get('top_left_image', None)
        top_left_reddit = template_metadata.get('top_left_reddit', None)
        top_left_template = template_metadata.get('top_left_template', None)
        if not top_left_image:
            top_left_image = Coordinate(template_metadata['x'], template_metadata['y'], 'template').get_image_coordinates()
        if not top_left_reddit:
            top_left_reddit = Coordinate(template_metadata['x'], template_metadata['y'], 'template').get_reddit_coordinates()
        if not top_left_template:
            top_left_template = (template_metadata['x'], template_metadata['y'])
        top_left = Coordinate(top_left_image[0], top_left_image[1], 'image')
        width = template_metadata.get('width', 100)
        height = template_metadata.get('height', 100)
        return cls(event_name, section_name, top_left, width, height, correct_image)

    @classmethod
    def download_reference_sections(cls, url: str, event_name: str, save: bool = True) -> List['ReferenceSection']:
        response = requests.get(url)
        response.raise_for_status()
        metadata = response.json()

        ref_sections = []

        for index, template_metadata in enumerate(metadata['templates'], start=1):
            name = template_metadata.get('name', '')
            source = template_metadata['sources'][0] if 'sources' in template_metadata and len(
                template_metadata['sources']) > 0 else ''
            x, y = template_metadata['x'], template_metadata['y']
            top_left = Coordinate(x, y, 'template')
            width, height = template_metadata.get('width', None), template_metadata.get('height', None)
            if source.startswith('http'):
                image_response = requests.get(source, stream=True)
                image_response.raise_for_status()
                correct_image = Image.open(io.BytesIO(image_response.content))
                if correct_image.mode != 'RGBA':
                    image = correct_image.convert('RGBA')
                width, height = correct_image.size
                if save:  # Save the image file locally
                    image_path = os.path.join('canvas_data', event_name, 'reference_sections', name, f"{name}.png")
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)  # Ensure the directories exist
                    correct_image.save(image_path)
            else:
                correct_image = Image.open(source)
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
            ref_section = cls(event_name, name, top_left, width, height, correct_image)
            if save:
                ref_section.save()
            ref_sections.append(ref_section)

        return ref_sections


class Event:
    """Represents an event and contains related canvas images and reference sections."""
    def __init__(self, event_name: str):
        self.event_name = event_name
        self.canvas_dir = os.path.join('canvas_data', event_name, 'canvas')
        self.reference_section_dir = os.path.join('canvas_data', event_name, 'reference_sections')
        self.canvas_images = self._load_canvas_images()
        self.reference_sections = self._load_reference_sections()

    def _load_canvas_images(self) -> List[Canvas]:
        canvas_dirs = glob.glob(os.path.join(self.canvas_dir, '*'))
        canvas_images = []
        for dir in canvas_dirs:
            filename = os.path.join(dir, f"{os.path.basename(dir)}.png")
            if os.path.exists(filename):
                canvas_images.append(Canvas(filename, self))
        canvas_images.sort(key=lambda x: x.timestamp)
        return canvas_images

    def _load_reference_sections(self) -> List[ReferenceSection]:
        section_dirs = glob.glob(os.path.join(self.reference_section_dir, '*'))
        return {dir.split('/')[-1]: ReferenceSection.load(dir) for dir in section_dirs}

    def create_basic_timelapse(self, reference_section_name: str = None,
                               output_path='frames/', start_time: datetime = datetime.utcnow() - timedelta(hours=2),
                               end_time: datetime = datetime.utcnow(), coordinates_from: str = None):

        plt.rc('font', size=12)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('legend', fontsize=12)
        plt.rc('figure', titlesize=25)

        timelapse_id = str(uuid.uuid4()).split('-')[0]
        timelapse_dir = os.path.join('canvas_data', self.event_name, 'timelapses', timelapse_id)
        os.makedirs(timelapse_dir, exist_ok=True)

        frames_to_process = sorted([canvas_frame for canvas_frame in self.canvas_images if start_time < canvas_frame.timestamp], key=lambda x: x.timestamp)
        num_frames = len(frames_to_process)

        if num_frames == 0:
            print("No frames were loaded. Please increase the time interval.")
            return

        # Define reference section name
        if reference_section_name is None:
            reference_section_name = f"timelapse_{timelapse_id}"

        if reference_section_name not in self.reference_sections:
            # Use coordinates from a pre-existing reference section
            if coordinates_from and coordinates_from in self.reference_sections:
                pre_existing_section = self.reference_sections[coordinates_from]
                top_left_image = pre_existing_section.top_left
                width = pre_existing_section.width
                height = pre_existing_section.height
                frames_to_process[0].add_reference_section(reference_section_name, top_left_image, width, height)
            else:
                print("Error: Neither a reference section nor a set of coordinates were provided.")
                return

        reference_section = self.reference_sections[reference_section_name]

        # Remove all previous .jpg files in output directory
        files = glob.glob(os.path.join(output_path, '*.jpg'))
        for f in files:
            os.remove(f)

        # Preprocess pixel changes and timestamps
        previous_state = None
        pixel_changes_list = []
        wrong_pixel_list = []
        timestamps_list = []
        pixel_change_rate_list = [0]
        net_change_list = [0]

        print(frames_to_process)

        with tqdm(total=num_frames, desc="Preprocessing frames",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            for n, canvas_frame in enumerate(frames_to_process):
                canvas_frame.load_image()
                canvas_state = np.array(reference_section.extract_from_canvas(canvas_frame))

                if n == 0:
                    timestamps_list.append(canvas_frame.timestamp)
                    pixel_changes_list.append(0)
                    wrong_pixel_list.append(0)
                    previous_state = canvas_state
                    reference_image = reference_section.get_correct_image_as_numpy()
                else:
                    # vs previous state
                    pixel_change_mask = (previous_state != canvas_state)
                    num_pixel_changes = pixel_change_mask.sum()
                    pixel_changes_list.append(num_pixel_changes)

                    # Calculate pixel change rate (pixel per second)
                    time_diff = (canvas_frame.timestamp - timestamps_list[-1]).total_seconds()
                    pixel_change_rate = num_pixel_changes / time_diff if time_diff != 0 else 0
                    pixel_change_rate_list.append(pixel_change_rate)

                    # vs reference image
                    wrong_pixel_mask = (canvas_state != reference_image)
                    num_wrong_pixels = wrong_pixel_mask.sum()
                    wrong_pixel_list.append(num_wrong_pixels)

                    net_change = (wrong_pixel_list[-1] - wrong_pixel_list[-2])
                    net_change_list.append(net_change)

                    timestamps_list.append(canvas_frame.timestamp)
                    pbar.update()

                    previous_state = canvas_state

        max_pixel_changes = max(pixel_changes_list)
        max_wrong_pixel_changes = max(wrong_pixel_list)
        max_pixel_change_rate = max(pixel_change_rate_list)
        max_net_change = max(abs(min(net_change_list)), abs(max(net_change_list)))

        with tqdm(total=num_frames, desc="Rendering frames",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            for n, canvas_frame in enumerate(frames_to_process):
                try:
                    #print(canvas_frame.image)
                    reference_section_state = reference_section.get_correct_image_as_numpy()
                    canvas_state = np.array(reference_section.extract_from_canvas(canvas_frame))

                    pixel_change_mask = np.all(reference_section_state == canvas_state, axis=-1)
                    pixel_changes = canvas_state.copy()
                    pixel_changes[pixel_change_mask] = 0

                    fig = plt.figure(figsize=(12, 12))
                    gs = GridSpec(4, 3, figure=fig)

                    ax1 = fig.add_subplot(gs[slice(0, 2), 0])
                    ax2 = fig.add_subplot(gs[slice(0, 2), 1])
                    ax3 = fig.add_subplot(gs[slice(0, 2), 2])

                    ax4 = fig.add_subplot(gs[2, :])
                    ax5 = fig.add_subplot(gs[3, :])

                    ax4.title.set_text('Pixel Change Rate (vs previous frame) per Second')
                    ax5.title.set_text('Pixel Errors (vs Reference Image)')

                    # Plot the number of pixel changes over time
                    ax4.plot(timestamps_list[:n + 1], pixel_change_rate_list[:n + 1], color='blue', lw=5)
                    ax5.plot(timestamps_list[:n + 1], wrong_pixel_list[:n + 1], color='blue', lw=5)

                    ax4.set_xlim([start_time, end_time])
                    ax5.set_xlim([start_time, end_time])
                    ax4.set_ylim([0, max_pixel_change_rate])
                    ax5.set_ylim([0, max_wrong_pixel_changes])

                    ax1.imshow(reference_section_state)
                    ax1.title.set_text('Reference Image')

                    ax2.imshow(canvas_state)
                    ax2.title.set_text(f'{canvas_frame.timestamp} UTC')

                    ax3.imshow(pixel_changes)
                    ax3.title.set_text('Diff')
                    plt.tight_layout()

                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

                    ax4.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 10, 20, 30, 40, 50]))
                    ax5.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 10, 20, 30, 40, 50]))

                    plt.gcf().autofmt_xdate()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_path, f'frame{n}.jpg'))
                    plt.close('all')
                    pbar.update()

                except FileNotFoundError:
                    print(f"File not found for timestamp: {canvas_frame.timestamp}")
                except Exception as e:
                    print(f"An error occurred while processing frame {n}: {e}")

        # Run the shell script after all frames have been rendered
        try:
            print("Starting rendering with render.sh...")
            subprocess.check_call(["bash", f"{output_path}/render.sh"])
            print("Rendering completed. The output file should be located in the frames/ directory.")

            # Copy the mp4 file, reference section, and JSON to the new directory
            shutil.copy(f"{output_path}/output.mp4", timelapse_dir)  # Assuming the rendered file is named "output.mp4"
            shutil.copy(
                os.path.join(self.reference_section_dir, reference_section_name, f"{reference_section_name}.png"),
                timelapse_dir)
            shutil.copy(
                os.path.join(self.reference_section_dir, reference_section_name, f"{reference_section_name}.json"),
                timelapse_dir)

            metadata = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "num_frames": num_frames
            }
            with open(os.path.join(timelapse_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f)

        except subprocess.CalledProcessError:
            print("An error occurred while trying to run the shell script.")

    def generate_change_history_image(self, reference_section_name: str, threshold_time: timedelta):
        """Generate an image where each pixel color indicates how recently it was last changed."""
        # Step 1: Load the most recent canvas instance
        current_canvas = self.canvas_images[-1]
        current_canvas.load_image()
        current_section = self.reference_sections[reference_section_name]
        current_state = np.array(current_section.extract_from_canvas(current_canvas))

        # Initialize a change history array with the same size as the current state
        change_history = np.full(current_state.shape[:2], np.inf)  # Initialize with infinity (no change yet)

        # Select canvases within the desired time range
        threshold_timestamp = current_canvas.timestamp - threshold_time
        relevant_canvases = [canvas for canvas in reversed(self.canvas_images[:-1]) if
                             canvas.timestamp >= threshold_timestamp]

        # Step 2: Iterate backward through the canvas history
        for canvas in tqdm(relevant_canvases, desc="Processing canvases", total=len(relevant_canvases)):
            canvas.load_image()
            past_state = np.array(current_section.extract_from_canvas(canvas))

            # Step 3: Pixel-wise comparison
            change_mask = np.any(past_state != current_state, axis=-1)

            # Step 4: Maintain a history of changes
            change_history[np.where(change_mask)] = (current_canvas.timestamp - canvas.timestamp).total_seconds()

            # Step 5: Terminate the search
            if np.all(change_history < np.inf):
                break

            current_state = past_state
            current_canvas = canvas

        # Normalize change_history to grayscale range
        max_time = change_history[change_history < np.inf].max()  # Get maximum non-infinity value
        change_history[change_history == np.inf] = max_time  # Replace infinities with maximum time
        change_history_normalized = (255 * (1.0 - change_history / max_time)).astype(
            np.uint8)  # Normalize to [0, 255] and invert

        # Normalize change_history to grayscale range
        max_time = change_history[change_history < np.inf].max()  # Get maximum non-infinity value
        change_history[change_history == np.inf] = 0  # Most recent changes are now represented as 0
        change_history_normalized = (255 * (1.0 - change_history / max_time)).astype(
            np.uint8)  # Normalize to [0, 255] and invert

        # Normalize change_history to grayscale range
        max_time = change_history[change_history < np.inf].max()  # Get maximum non-infinity value
        change_history[change_history == np.inf] = 0  # Most recent changes are now represented as 0
        change_history_normalized = (255 * (1.0 - change_history / max_time)).astype(
            np.uint8)  # Normalize to [0, 255] and invert

        # Step 6: Normalize and visualize the results
        history_image = Image.fromarray(change_history_normalized)

        # Display the heatmap and reference section side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Display reference section
        reference_image = np.array(current_section.get_correct_image_as_numpy())
        ax1.imshow(reference_image)
        ax1.set_title("Reference Section")

        # Display heatmap
        im = ax2.imshow(change_history, cmap='magma_r', interpolation='nearest', vmin=0, vmax=max_time)
        start_time_str = current_canvas.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ax2.set_title(f"Change History Heatmap\n(start time: {start_time_str}, search range: {str(threshold_time)})")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax2)

        def format_func(value, tick_number):
            """Convert number of seconds to 'mm:ss' format."""
            return f'{value // 60:.0f}:{value % 60:02.0f}'

        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        cbar.set_label('Time since last change')

        plt.tight_layout()
        plt.show()

        return history_image