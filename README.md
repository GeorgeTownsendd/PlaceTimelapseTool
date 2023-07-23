# PlaceTimelapseTool

PlaceTimelapseTool is an advanced utility designed to record and analyze the evolution of a Reddit collaborative canvas over time. It utilizes Reddit's API to capture the real-time changes on the canvas, storing the snapshots as a sequence of images. Additionally, it offers functionality for producing time-lapse videos and analyzing specific reference sections.

## Directory Structure

```
PlaceTimelapseTool
| - canvas_data
   | - place_2023
     | - canvas
     | - reference_sections
       | - section1
       | - section2
     | - timelapses
| - canvas_analysis.py
| - canvas_archiver.py
| - authparams.py
```

## Usage

### Canvas

`Canvas` class represents a snapshot of the canvas at a particular time. 

```python
from canvas_analysis import Canvas
canvas = Canvas(filename, event, load_image)
```

### ReferenceSection

`ReferenceSection` instances define areas of interest on the canvas for specific analysis. You can download overlay templates into a ReferenceSection for precise tracing.

```python
from canvas_analysis import ReferenceSection
reference_section = ReferenceSection(event_name, name, top_left, width, height, correct_image)
```

### Event

`Event` instances represent unique events associated with a series of canvas images and reference sections.

```python
from canvas_analysis import Event
event = Event("place_2023")
```

### Timelapse Creation

Generate time-lapse videos using the `create_basic_timelapse` function from the `Event` class.

```python
event.create_basic_timelapse(reference_section_name, start_time=start_time, end_time=end_time)
```

## Getting Started

Create an `authparams.py` file to include your Reddit API credentials (username, password, OAuth client, OAuth secret).

To run the main archiving script, execute:

```
python canvas_archiver.py
```
