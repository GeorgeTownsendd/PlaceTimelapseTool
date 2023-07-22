import os
import json
import shutil
import glob
from typing import List
from PIL import Image
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import subprocess
import uuid

def reddit_to_image_coordinates(coord: tuple) -> tuple:
    center = (1500, 1000)
    return coord[0] + center[0], coord[1] + center[1]


def image_to_reddit_coordinates(coord: tuple) -> tuple:
    center = (1500, 1000)
    return coord[0] - center[0], coord[1] - center[1]


class Canvas:
    def __init__(self, filename: str, event, load_image: bool = False):
        self.filename = filename
        self.event = event
        self.image = Image.open(filename) if load_image else None
        self.timestamp = self._extract_timestamp_from_filename()

    def _extract_timestamp_from_filename(self):
        # Extract timestamp from filename and convert to datetime
        timestamp_str = os.path.basename(self.filename).split('.')[0]
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

    def load_image(self):
        # Load the image if it has not been loaded
        if self.image is None:
            self.image = Image.open(self.filename)

    def add_reference_section(self, section_name: str, top_left: tuple, width: int, height: int):
        # Create a reference section from this canvas
        reference_section = ReferenceSection.from_canvas(top_left, width, height, self)

        # Define the path where to save the reference section
        section_dir = os.path.join(self.event.reference_section_dir, section_name)
        os.makedirs(section_dir, exist_ok=True)

        # Save the reference section to disk and add it to the event's list of reference sections
        reference_section.save(section_dir, section_name)
        self.event.reference_sections[section_name] = reference_section



class Section:
    def __init__(self, top_left: tuple, width: int, height: int):
        self.top_left = top_left
        self.width = width
        self.height = height
        self.bottom_right = (top_left[0]+width, top_left[1]+height)  # Calculate bottom right coordinate of rectangle

    def extract_from_canvas(self, canvas: Canvas):
        # Load the image if it has not been loaded
        canvas.load_image()

        # Extract the section from the given canvas image using PIL's crop function
        # Return the cropped image
        return canvas.image.crop((*self.top_left, *self.bottom_right))


class ReferenceSection(Section):
    def __init__(self, top_left_image: tuple, width: int, height: int, correct_image: Image.Image):
        super().__init__(top_left_image, width, height)
        self.correct_image = correct_image
        self.top_left_reddit = image_to_reddit_coordinates(top_left_image)

    def get_correct_image_as_numpy(self) -> np.ndarray:
        # Convert the correct_image to a NumPy array
        return np.array(self.correct_image)

    @classmethod
    def from_canvas(cls, top_left_image: tuple, width: int, height: int, canvas: Canvas):
        canvas.load_image()  # Ensure the image is loaded
        # Calculate bottom right coordinates for cropping
        bottom_right = (top_left_image[0] + width, top_left_image[1] + height)
        section = cls(top_left_image, width, height, canvas.image.crop((*top_left_image, *bottom_right)))
        return section

    def save(self, path: str, section_name: str):
        self.correct_image.save(os.path.join(path, f"{section_name}.png"), "PNG")
        with open(os.path.join(path, f"{section_name}.json"), 'w') as f:
            json.dump({
                'top_left_image': self.top_left,
                'width': self.width,
                'height': self.height
            }, f)

    @classmethod
    def load(cls, path: str) -> 'ReferenceSection':
        section_name = os.path.basename(path)
        correct_image = Image.open(os.path.join(path, f"{section_name}.png"))
        with open(os.path.join(path, f"{section_name}.json"), 'r') as f:
            metadata = json.load(f)
        top_left_image = metadata['top_left_image']
        return cls(top_left_image, metadata['width'], metadata['height'], correct_image)


class Event:
    def __init__(self, event_name: str):
        self.event_name = event_name
        self.canvas_dir = os.path.join('canvas_data', event_name, 'canvas')
        self.reference_section_dir = os.path.join('canvas_data', event_name, 'reference_sections')
        self.canvas_images = self._load_canvas_images()
        self.reference_sections = self._load_reference_sections()

    def _load_canvas_images(self) -> List[Canvas]:
        # Load all canvas images in the directory
        canvas_dirs = glob.glob(os.path.join(self.canvas_dir, '*'))
        return [Canvas(os.path.join(dir, f"{os.path.basename(dir)}.png"), self) for dir in canvas_dirs]

    def _load_reference_sections(self) -> List[ReferenceSection]:
        # Load all reference sections in the directory
        section_dirs = glob.glob(os.path.join(self.reference_section_dir, '*'))
        return {dir.split('/')[-1]: ReferenceSection.load(dir) for dir in section_dirs}

    def analyze(self):
        # Analyze the event's canvas images and reference sections
        pass  # You need to implement this function

    def create_basic_timelapse(self, reference_section_name: str = None, basic_version=False,
                               output_path='frames/', start_time: datetime = datetime.utcnow() - timedelta(hours=2),
                               end_time: datetime = datetime.utcnow(), coordinates_from: str = None):

        # Generate a unique ID for the timelapse
        timelapse_id = str(uuid.uuid4())
        timelapse_dir = os.path.join('canvas_data', self.event_name, 'timelapses', timelapse_id)
        os.makedirs(timelapse_dir, exist_ok=True)

        frames_to_process = sorted([canvas_frame for canvas_frame in self.canvas_images
                                    if start_time < canvas_frame.timestamp < end_time],
                                   key=lambda x: x.timestamp)
        num_frames = len(frames_to_process)

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
                return  # No reference section was created, so return without creating a timelapse

        reference_section = self.reference_sections[reference_section_name]

        # Remove all previous .jpg files in output directory
        files = glob.glob(os.path.join(output_path, '*.jpg'))
        for f in files:
            os.remove(f)

        frames_to_process = sorted([canvas_frame for canvas_frame in self.canvas_images
                                    if start_time < canvas_frame.timestamp < end_time],
                                   key=lambda x: x.timestamp)
        num_frames = len(frames_to_process)

        with tqdm(total=num_frames, desc="Processing frames",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            for n, canvas_frame in enumerate(frames_to_process):
                try:
                    reference_section_state = reference_section.get_correct_image_as_numpy()
                    canvas_frame.load_image()
                    canvas_state = np.array(reference_section.extract_from_canvas(canvas_frame))
                    pixel_change_mask = (reference_section_state == canvas_state)
                    pixel_changes = np.zeros(reference_section_state.shape)
                    pixel_changes[pixel_change_mask] = 1

                    fig = plt.figure(figsize=(12, 12))

                    gs = GridSpec(4, 3, figure=fig)  # Changed GridSpec to a 6x3 grid

                    if basic_version:
                        ax1 = fig.add_subplot(gs[0, 0])
                        ax2 = fig.add_subplot(gs[0, 1])
                        ax3 = fig.add_subplot(gs[0, 2])
                    else:
                        ax1 = fig.add_subplot(gs[slice(0, 2), 0])
                        ax2 = fig.add_subplot(gs[slice(0, 2), 1])
                        ax3 = fig.add_subplot(gs[slice(0, 2), 2])

                        ax4 = fig.add_subplot(gs[2, :])  # Middle row, one plot
                        ax5 = fig.add_subplot(gs[3, :])  # Bottom row, one plot

                        ax4.title.set_text('Additional Plot - Future Implementation')
                        ax5.title.set_text('Additional Plot - Future Implementation')

                    ax1.imshow(reference_section_state)
                    ax1.title.set_text('Reference Image')

                    ax2.imshow(canvas_state)
                    ax2.title.set_text(f'Canvas State - {canvas_frame.timestamp}')

                    ax3.imshow(pixel_changes, cmap='gray')
                    ax3.title.set_text('Diff')

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

            # Create the metadata JSON
            metadata = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "num_frames": num_frames
            }
            with open(os.path.join(timelapse_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f)

        except subprocess.CalledProcessError:
            print("An error occurred while trying to run the shell script.")

if __name__ == "__main__":
    #top_left_reddit = (220, 120)
    top_left_reddit = (190, 135)
    top_left_image = reddit_to_image_coordinates(top_left_reddit)
    event = Event("place_2023")

    event.create_basic_timelapse(coordinates_from='NormalHair', start_time=datetime.utcnow() - timedelta(hours=2))

    #create_basic_timelapse(event, event.reference_sections['NormalHair'])
    #canvas = event.canvas_images[1900]
    #canvas.add_reference_section('NormalHair', top_left_image, 60, 60)
