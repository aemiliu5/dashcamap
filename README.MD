# dashcamap

Python script that generates a map from hardcoded GPS data dashcam video using Optical Character Recognition.

![image](https://user-images.githubusercontent.com/21015074/227303585-77e2029c-bfd1-4363-94c8-e68f1a80700e.png)
![image](https://user-images.githubusercontent.com/21015074/227303273-bf346c98-b4a1-4750-b667-7870bdc32b39.png)

## Process

Upon running dashcamap with a video file, it will:

* Extract video frames with GPS data (usually once every second, this is configurable).
* Pre-process them by grayscaling and thresholding the extracted images.
* Perform OCR on the frames and extract the GPS data.
* Clean-up any inconsistencies in the GPS data found.
* Produce a file with a map containing the extracted GPS data as markers & lines.

## Requirements
* Python 3
* OpenCV
* Tesseract OCR
* Numpy
* Folium

You can install all requirements by running:
```
pip install -r requirements.txt
```

Aside from the python requirements, you will also need to install Tesseract OCR. For Windows users, you can download the executable [here](https://github.com/UB-Mannheim/tesseract/wiki).

For Linux and macOS users, check your respective package managers.

## Usage

```
py dashcamap.py <filename> [arguments]
```
Possible arguments:
```
-c / --clear          Clears all generated files apart from the map on finish.
-f / --frame          Prints verbose frame data.
-t / --threshold      Cleanup difference threshold.
-v / --verbose        Verbose console logging.
--flush               Delete leftover data from previous runs.
--fps                 Set framerate (default 30)
--skip-extract        Skips extraction step (uses existing generated frames on "frames" folder)
--skip-process        Skips processing step (uses existing generated frames on "frames2" folder)
--skip-cleanup        Skips cleanup step (may result in data with errors)
--simple-markers      Include only start and end markers.
--no-markers          Generates a map with just a line and no markers
--set-crop            Sets cropping parameters. (default is for a VIOFO A119)
--preview-crop        Generate a preview window with a single cropped frame.

```

## License
MIT License
