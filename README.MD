# dashcamap
Python script that generates a map from hardcoded GPS data dashcam video using Optical Character Recognition.

## Usage
```
py dashcamap.py <filename> [arguments]
```

Possible arguments:
```
-c/--clear		Clears all generated files apart from the map on finish.
-f/--frame		Prints verbose frame data.
-t/--threshold	Cleanup difference threshold.
-v/--verbose	Verbose console logging.
--flush			Delete leftover data from previous runs.
--fps			Set framerate (default 30)
--skip-extract	Skips extraction step (uses existing generated frames on "frames" folder)
--skip-process	Skips processing step (uses existing generated frames on "frames2" folder)
--skip-cleanup	Skips cleanup step (may result in data with errors)
--no-markers	Generates a map with just a line and no markers
--set-crop		Sets cropping parameters. (default is for a VIOFO A119 2560x1440)
```

## License
MIT License
