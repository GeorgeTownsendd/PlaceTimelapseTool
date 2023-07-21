import os
import json
import shutil
import glob
from typing import List
from PIL import Image
from datetime import datetime


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
        self.event.reference_sections.append(reference_section)



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
        return [ReferenceSection.load(dir) for dir in section_dirs]

    def analyze(self):
        # Analyze the event's canvas images and reference sections
        pass  # You need to implement this function

    def prepare_timelapse(self, output_path: str, target_fps: int):
        total_time = (self.canvas_images[-1].timestamp - self.canvas_images[0].timestamp).total_seconds()
        ideal_frame_interval = total_time / target_fps

        next_frame_time = self.canvas_images[0].timestamp
        for image in self.canvas_images:
            if (image.timestamp - next_frame_time).total_seconds() >= ideal_frame_interval:
                shutil.copy(image.filename, output_path)  # Copy the selected images to the output path
                next_frame_time = image.timestamp


if __name__ == "__main__":
    top_left_reddit = (227, 129)
    top_left_image = reddit_to_image_coordinates(top_left_reddit)
    event = Event("place_2023")
    canvas = event.canvas_images[0]
    canvas.add_reference_section('test', top_left_image, 100, 100)
