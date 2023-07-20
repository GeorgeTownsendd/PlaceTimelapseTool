import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime, timedelta
from PIL import Image
import json
import bisect


def find_closest_datetime_index(sorted_list, input_datetime):
    index = bisect.bisect_left(sorted_list, input_datetime)
    if index == 0:
        return index
    if index == len(sorted_list):
        return index - 1
    before = sorted_list[index - 1]
    after = sorted_list[index]
    if after - input_datetime < input_datetime - before:
        return index
    else:
        return index - 1


class SubCanvas:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        split_directory = self.base_dir.split('/')
        self.event_name = split_directory[-2]
        self.subcanvas_gridloc = split_directory[-1][-2:]

        self.image_list = os.listdir(self.base_dir)
        self.image_timestamps = [datetime.strptime(os.path.splitext(os.path.basename(file))[0], '%Y-%m-%d %H:%M:%S.%f') for file in self.image_list]

        self.sorted_image_list = [x for y, x in sorted(zip(self.image_timestamps, self.image_list))]
        self.sorted_image_timestamps = sorted(self.image_timestamps)

        self.in_memory = False
        self.loaded_image = np.zeros((1000, 1000, 3))

    def load_to_memory(self, timestamp='latest', window_seconds=0):
        if len(self.sorted_image_list) == 0: #subcanvas may not have any data/exist
            self.loaded_image_filename = 'NoFile'
            return np.zeros((1000, 1000, 3))

        else:
            if timestamp == 'latest':
                self.loaded_image_filename = self.sorted_image_list[-1]
                self.loaded_image = self.load_image(self.loaded_image_filename)

            elif window_seconds != 0:
                closet_image_timestamp_index = find_closest_datetime_index(self.sorted_image_timestamps, timestamp)

                if window_seconds > 0:
                    if (self.sorted_image_timestamps[
                            closet_image_timestamp_index] - timestamp).seconds < window_seconds:
                        self.loaded_image_filename = self.sorted_image_list[closet_image_timestamp_index]
                        self.loaded_image = self.load_image(self.loaded_image_filename)
                    else:
                        print('No SubCanvas images within threshold')

            else:
                closet_image_timestamp_index = find_closest_datetime_index(self.sorted_image_timestamps, timestamp)
                self.loaded_image = self.loaded_image(closet_image_timestamp_index)

    def get_image_nearest_timestamp(self, timestamp, window_seconds=0):
        closet_image_timestamp_index = find_closest_datetime_index(self.sorted_image_timestamps, timestamp)

        return self.load_image(self.sorted_image_list[closet_image_timestamp_index])


    def load_image(self, img_path):
        img = Image.open(os.path.join(self.base_dir, img_path))
        img = img.convert("RGB")
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        return img_array


    def generate_subcanvas_plot(self):
        if self.in_memory:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.loaded_image)
            ax.set_title(f'SubCanvas from {self.event_name} at grid location {self.subcanvas_gridloc}')
            return fig, ax
        else:
            print('No image has been loaded into memory.')
            return None, None

    def display(self):
        self.generate_subcanvas_plot()
        plt.show()


class Event:
    def __init__(self, event_name):
        self.event_name = event_name
        self.base_dir = os.path.join('canvas_data', self.event_name)

        if os.path.exists(self.base_dir):
            self.initalised = True
        else:
            self.initalised = False

        if self.initalised:
            pattern = re.compile('subcanvas_[0-9][0-9]')
            subcanvases_names = [folder_name[-2:] for folder_name in os.listdir(self.base_dir) if pattern.match(folder_name)]
            self.subcanvases = [SubCanvas(os.path.join(self.base_dir, f'subcanvas_{sc[0]}{sc[1]}')) for sc in subcanvases_names]

            self.event_image_list = []
            self.event_image_timestamp_list = []
            for subcanvas in self.subcanvases:
                self.event_image_list += subcanvas.image_list
                self.event_image_timestamp_list += subcanvas.image_timestamps

            self.sorted_event_image_list = [x for y, x in sorted(zip(self.event_image_timestamp_list, self.event_image_list))]
            self.sorted_event_image_timestamp_list = sorted(self.event_image_timestamp_list)

            self.data_start = self.sorted_event_image_timestamp_list[0]
            self.data_end = self.sorted_event_image_timestamp_list[1]


class FrozenCanvas:
    def __init__(self, event, start_time, window_size_seconds, grid_size=1):
        self.event = event
        self.event_name = self.event.event_name
        self.grid_size = grid_size
        self.canvas_size = self.grid_size * 1000
        self.start_time = start_time

        canvas_image = np.zeros((self.canvas_size, self.canvas_size, 3))
        for subcanvas in self.event.subcanvases:
            if not subcanvas.in_memory:
                subcanvas.load_to_memory(timestamp=start_time, window_seconds=window_size_seconds)

            subcanvas_id = subcanvas.subcanvas_gridloc
            subcanvas_image = subcanvas.loaded_image

            x_start = (int(subcanvas_id[0]) % self.grid_size) * 1000
            y_start = (int(subcanvas_id[1]) % self.grid_size) * 1000

            canvas_image[y_start:y_start+1000, x_start:x_start+1000] = subcanvas_image

        self.canvas_image = canvas_image

    def generate_frozen_canvas_plot(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.canvas_image, origin='upper')
        ax.set_title(f'FrozenCanvas from {self.event_name}')
        return fig, ax

    def display(self):
        self.generate_frozen_canvas_plot()
        plt.show()

    def create_and_save_reference_section(self, ref_name, top_left, width, height):
        x, y = top_left
        reference_image = self.canvas_image[y:y+height, x:x+width]

        reference_dir = os.path.join('canvas_data', self.event_name, 'reference_sections')
        os.makedirs(reference_dir, exist_ok=True)

        reference_image = Image.fromarray((reference_image * 255).astype(np.uint8))
        reference_image_path = os.path.join(reference_dir, f'{ref_name}.png')
        reference_image.save(reference_image_path)

        metadata = {
            'top_left': top_left,
            'width': width,
            'height': height
        }
        with open(os.path.join(reference_dir, f'{ref_name}_metadata.json'), 'w') as json_file:
            json.dump(metadata, json_file, indent=4)


class Section:
    def __init__(self, top_left, width, height):
        self.top_left = top_left
        self.width = width
        self.height = height


class ReferenceSection(Section):
    def __init__(self, event_name, ref_name):
        self.event_name = event_name
        self.load_reference(ref_name)

    def save_reference(self, ref_name):
        reference_dir = os.path.join('canvas_data', self.event_name, 'reference_sections')
        os.makedirs(reference_dir, exist_ok=True)

        reference_image = Image.fromarray((self.reference_image * 255).astype(np.uint8))
        reference_image_path = os.path.join(reference_dir, f'{ref_name}.png')
        reference_image.save(reference_image_path)

        metadata = {
            'top_left': self.top_left,
            'width': self.width,
            'height': self.height
        }
        with open(os.path.join(reference_dir, f'{ref_name}_metadata.json'), 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

    def load_reference(self, ref_name):
        reference_dir = os.path.join('canvas_data', self.event_name, 'reference_sections')
        metadata_file = os.path.join(reference_dir, f'{ref_name}_metadata.json')
        with open(metadata_file, 'r') as json_file:
            metadata = json.load(json_file)

        self.top_left = tuple(metadata['top_left'])
        self.width = metadata['width']
        self.height = metadata['height']

        reference_image_path = os.path.join(reference_dir, f'{ref_name}.png')
        self.reference_image = self.load_image(reference_image_path)

    def compare_with_frozen_canvas(self, frozen_canvas):
        x, y = self.top_left
        comparison_image = frozen_canvas.canvas_image[y:y+self.height, x:x+self.width]

        difference = np.where(np.all(self.reference_image == comparison_image, axis=-1), 0, 1)
        return difference

    def display_comparison(self, frozen_canvas):
        difference = self.compare_with_frozen_canvas(frozen_canvas)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Show Reference Image
        axs[0].imshow(self.reference_image, origin='upper')
        axs[0].set_title("Reference Image", fontsize=20)
        self.set_labels(axs[0])

        # Show Frozen Canvas Image
        x, y = self.top_left
        comparison_image = frozen_canvas.canvas_image[y:y+self.height, x:x+self.width]
        axs[1].imshow(comparison_image, origin='upper')
        axs[1].set_title(str(frozen_canvas.start_time), fontsize=20)
        self.set_labels(axs[1])

        # Show Binary Change Map
        axs[2].imshow(difference, cmap='gray', origin='upper')
        axs[2].set_title("Wrong Pixels", fontsize=20)
        self.set_labels(axs[2])

        plt.tight_layout()
        plt.show()

    def set_labels(self, ax):
        x_ticks = np.arange(0, self.width, 5)
        y_ticks = np.arange(0, self.height, 5)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_ticks + self.top_left[0])
        ax.set_yticklabels(y_ticks + self.top_left[1])

    @staticmethod
    def load_image(img_path):
        img = Image.open(img_path)
        img = np.array(img) / 255.0
        return img


if __name__ == '__main__':
    reference_section = ReferenceSection('place_2022', 'HI_test')

    event = Event('place_2022')
    frozen_canvas = FrozenCanvas(event, datetime(2022, 4, 2, 23, 18, 46), window_size_seconds=15)

    x = reference_section.display_comparison(frozen_canvas)

    #frozen_canvas.display()