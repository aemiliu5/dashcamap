import sys
import os
import re
import shutil
import cv2
import numpy as np
import folium
import argparse
import pytesseract

from os import path

# Pytesseract Initialization & Config
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
custom_config = r'-c tessedit_char_whitelist="ENSW.KM/PHenswkmh0123456789 " --oem 3 --psm 7 -l eng'

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='dashcamap.py - A Python script that extracts GPS data from dashcam videos using OCR.')

parser.add_argument('-c', '--clear', 	action='store_true', 	help='Clears all generated data on finish.')
parser.add_argument('-f', '--frame', 	action='store_true', 	help='Print data per frame file.')
parser.add_argument('-t', '--threshold',type=float, 		default=0.2, help='Cleanup difference threshold.')
parser.add_argument('-v', '--verbose', 	action='store_true', 	help='Verbose messages.')
parser.add_argument('--skip-extract', 	action='store_true', 	help='Skip extraction step and use existing generated frames.')
parser.add_argument('--skip-process', 	action='store_true', 	help='Skip processing step and use existing generated frames.')
parser.add_argument('--skip-cleanup', 	action='store_true', 	help='Skip cleanup step of coordinates.')
parser.add_argument('--simple-markers', action='store_true', 	help="Don't include inbetween markers in the generated map.")
parser.add_argument('--no-markers', 	action='store_true', 	help="Don't include any markers in the generated map.")
parser.add_argument('--set-crop', 		type=int, default=[1380,1440,20,620], nargs=4, help="Define cropping area")
parser.add_argument('--preview-crop', 	action='store_true', 	help="Preview cropping area")
parser.add_argument('--flush', 			action='store_true', 	help='Flush leftover data.')
parser.add_argument('--fps', 			type=int, default=30,	help='Framerate of video to check (GPS updates usually every second)')
parser.add_argument('file', 			nargs='?', default=None, type=str, help="File")

args = parser.parse_args()

# --- FLUSH DATA ---
if args.flush:
	print("Flushing previously generated data...")
	if path.exists('./frames'): 			shutil.rmtree('./frames')
	if path.exists('./frames2'): 			shutil.rmtree('./frames2')
	if path.exists('recognized.txt'): 		os.remove('recognized.txt')
	if path.exists('recognized2.txt'):		os.remove('recognized2.txt')
	if path.exists('recognized3.txt'):		os.remove('recognized3.txt')
	if path.exists('map.html'): 			os.remove('map.html')
	print("Files successfully deleted.")
	quit()

# --- PREVIEW CROP ---
if args.preview_crop:
	video = cv2.VideoCapture(args.file)
	
	if not video.isOpened():
		raise Exception("Error opening video file.")
	
	ret, frame = video.read()

	if ret == True:
		crop = frame[args.set_crop[0]:args.set_crop[1], args.set_crop[2]:args.set_crop[3]] 
		cv2.imshow("Crop preview", crop)
	
	cv2.waitKey()
	video.release()
	cv2.destroyAllWindows()
	quit()

# --- VARIABLES ---
regexCoords = re.compile(r"\b\d{1,3}\s(KM/H|MPH)\s[NS]\d{1,2}\.\d{6}\s[EW]\d{1,3}\.\d{6}\b")
frameCount = 0

# ----- FRAME EXTRACTION -----
# Extract frames that include GPS info. 
# Most dashcams update GPS info every second, so the program skips a set amount of frames to account for that (default 30)
if not (args.skip_extract or args.skip_process):
	video = cv2.VideoCapture(args.file)

	if not video.isOpened():
		raise Exception("Error opening video file.")

	if not os.path.exists('frames'):
		os.makedirs('frames')

	print("Extracting frames from " + args.file)
	
	while video.isOpened():
		ret, frame = video.read()

		if ret == True:
			crop = frame[args.set_crop[0]:args.set_crop[1], args.set_crop[2]:args.set_crop[3]] 
			cv2.imwrite('frames/frame{:d}.jpg'.format(frameCount), crop)
			frameCount += 1
			video.set(cv2.CAP_PROP_POS_FRAMES, frameCount * args.fps)

			if args.verbose:
				print(f"Extracting frame {frameCount}")
		else:
			break

	totalFrameCount = frameCount

	video.release()
	cv2.destroyAllWindows()
	print(f"Extracted {totalFrameCount} frames.")
else:
	totalFrameCount = len(os.listdir('frames'))
	print(f"Found {totalFrameCount} frames.")

frameCount = 0

# ----- FRAME PROCESSING -----
# Apply various techniques to make the image clearer to extract data from.
if not args.skip_process:
	file = open("recognized.txt", "a+")

	if args.frame:
		file3 = open("recognized3.txt", "a+")

	print("Processing frames...")
	while frameCount < totalFrameCount:

		if not os.path.exists('frames2'):
			os.makedirs('frames2')

		# Import image
		img = cv2.imread(f'frames/frame{frameCount}.jpg')

		if args.verbose:
			print(f"Processing frame {frameCount}")

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

		final = threshed

		# Perform OCR
		data = pytesseract.image_to_string(final, lang='eng', config=custom_config)
		file.write(f"{data}")
		
		if args.frame:
			file3.write(f"{frameCount}: {data}")

		cv2.imwrite(f'frames2/frame{frameCount}.jpg', final)
		frameCount += 1

	print(f"Processed {totalFrameCount} frames.")

# ----- IMAGE ANALYSIS -----
# Perform OCR on the processed images to extract GPS coordinates.
print("Analyzing frames...")

lat = 0.0
lon = 0.0
coords = []

if path.exists('recognized.txt'):
	with open("recognized.txt", "r") as file:
		lines = file.readlines()

	with open("recognized2.txt", "a+") as file2:
		for line in lines:
			match = regexCoords.search(line)
			
			if match:
				regexLat = re.search(r"[NS](\d+\.\s*\d+)", line)
				regexLon = re.search(r"[EW](\d+\.\s*\d+)", line)
				
				if regexLat:
					if 'N' in line:
						lat = float(regexLat.group(1).replace(" ", ""))
					else:
						lat = float('-' + regexLat.group(1).replace(" ", ""))

				if regexLon:
					if 'E' in line:
						lon = float(regexLon.group(1).replace(" ", ""))
					else:
						lon = float('-' + regexLon.group(1).replace(" ", ""))

				if regexLat and regexLon:
					coords.append([lat, lon])

				file2.write(match.group() + "\n")

	file.close()
	file2.close()

# ----- COORDINATES CLEANUP -----
# Scans all analyzed coordinates for possible irregularities and if found, removes them.
if not args.skip_cleanup:
	cleanup = []

	for i in range(1, len(coords)):
		lat1, lon1 = coords[i-1]
		lat2, lon2 = coords[i]
		diff_lat = abs(lat1 - lat2)
		diff_lon = abs(lon1 - lon2)

		if args.verbose:
			print(diff_lat, diff_lon)

		if diff_lat > args.threshold or diff_lon > args.threshold:
			cleanup.append(coords[i])

	for k in cleanup:
		coords.remove(k)

	print(f"Analyzed frames, extracted {len(coords)} coords, cleaned up {len(cleanup)} coords.")
else:
	print(f"Analyzed frames, extracted {len(coords)} coords.")

# ----- MAP GENERATION -----
# Generates a map of the route using the extracted coordinates.

if len(coords) > 0:
	print("Generating map...")
	map = folium.Map(location=[lat, lon])

	if not args.no_markers:
		for c in coords:
			if c == coords[0]: 						folium.Marker(location=c, popup='Start', icon=folium.Icon(icon='hand-point-up',  prefix='fa', color='green')).add_to(map)
			elif c == coords[len(coords)-1]:		folium.Marker(location=c, popup='End', 	 icon=folium.Icon(icon='flag-checkered', prefix='fa', color='red')).add_to(map)
			elif not args.simple_markers:			folium.Marker(c).add_to(map)

	folium.PolyLine(coords).add_to(map)
	map.save("map.html")

	print("Map generated.")
else:
	print("A map could not be generated as the coordinates could not be read. Please adjust the settings or try a different file.")

if args.clear:
	print("Deleting generated data...")
	if path.exists('./frames'): 			shutil.rmtree('./frames')
	if path.exists('./frames2'): 			shutil.rmtree('./frames2')
	if path.exists('recognized.txt'): 		os.remove('recognized.txt')
	if path.exists('recognized2.txt'):		os.remove('recognized2.txt')
	if path.exists('recognized3.txt'):		os.remove('recognized3.txt')
	print("Files successfully deleted.")

print("Finished.")