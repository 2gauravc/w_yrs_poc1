import numpy as np
import pandas as pd
import cv2
import os
import re
import zipfile
import logging

import matplotlib.pyplot as plt
import seaborn as sns

class VJump_data():

	def __init__(self, video_path):

		self.video_path = video_path

	def create_frame_path(self, frame_counter):
		vid_path, vid_name_ext = os.path.split(self.video_path)
		vid_name, vid_ext = os.path.splitext(vid_name_ext)

		frame_path = os.path.join(vid_path, vid_name)
		
		if not os.path.exists(frame_path):
			os.makedirs(frame_path)
			
		frame_path = os.path.join(frame_path, f"{vid_name}_FRAME_{frame_counter}.png")

		return frame_path

	def count_and_save_frames(self):

		# Source 1: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
		# Source 2: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

		# Creating a VideoCapture object.
		cap = cv2.VideoCapture(self.video_path)

		# Getting the video frame width and height.
		self.frame_width = int(cap.get(3))
		self.frame_height = int(cap.get(4))
		
		total_frames = 0

		while(cap.isOpened()):

			# Grabbing each individual frame, frame-by-frame.
			ret, frame = cap.read()

			if ret==True:

				total_frames += 1

				cv2.imwrite(self.create_frame_path(total_frames), frame)

				# Exiting if "Q" key is pressed on the keyboard.
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		# Releasing the VideoCapture object.
		cap.release()
		
		self.total_frames = total_frames

	def import_body_net(self):

		# Source for code: https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

		# Specify the paths for the 2 files
		protoFile = "../inputs_outputs/models/pose_deploy.prototxt"
		weightsFile = "../inputs_outputs/models/pose_iter_584000.caffemodel"

		# Read the network into Memory
		self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

	def find_skeleton(self, frame):

		frame_copy = np.copy(frame)

		# Specify the input image dimensions
		inWidth = 368
		inHeight = 368

		# Prepare the frame to be fed to the network
		inpBlob = cv2.dnn.blobFromImage(frame_copy, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

		# Set the prepared object as the input blob of the network
		self.net.setInput(inpBlob)

		output = self.net.forward()

		H = output.shape[2]
		W = output.shape[3]

		# Empty list to store the detected keypoints
		points = []
		
		for i in range(15):
			# Confidence map of corresponding body's part.
			probMap = output[0, i, :, :]

			# Find global maxima of the probMap.
			minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

			# Scale the point to fit on the original image
			x = (frame_copy.shape[1] * point[0]) / W
			y = (frame_copy.shape[0] * point[1]) / H

			if prob > 0.5:
				# Add the point to the list if the probability is greater than the threshold
				points.append((int(x), int(y)))

			else:
				points.append(None)
		
		return points
	
	def locate_body_points(self):
	
		# Creating a VideoCapture object.
		cap = cv2.VideoCapture(self.video_path)

		# Defining a new dataframe to store the skeleton point coordinates from each frame of the video.
		df_vid_points = pd.DataFrame()
		frame_counter = 1

		while(cap.isOpened()):

			# Grabbing each individual frame, frame-by-frame.
			ret, frame = cap.read()

			if ret==True:

				print(f"Analysing video - frame {frame_counter} of {self.total_frames}...")
				logging.debug(f"Locating body points - frame {frame_counter} of {self.total_frames}.")

				# Running human skeleton points detection on the grabbed frame.
				skeleton_frame_points = self.find_skeleton(frame)

				# Saving frame output skeleton point coordinates as a new row in above defined dataframe.
				df_vid_points[frame_counter] = skeleton_frame_points
				frame_counter += 1

				# Exiting if "Q" key is pressed on the keyboard.
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		# Releasing the VideoCapture object.
		cap.release()
		
		df_vid_points = df_vid_points.T.copy()
		
		self.df_vid_points = df_vid_points
		
		# self.df_vid_points_norm = self.df_vid_points.copy()

	def normalize_body_points(self):

		from VJump import Body_Points_Normalization

		self.df_vid_points_norm = Body_Points_Normalization.normalizing_body_points_in_df(self.df_vid_points, 1, 8, 3, 0)

	def separate_x_y_points(self, df):

		return_df = pd.DataFrame(index=df.index)

		for p in [col for col in df.columns if type(col)==int]:

			p_x = []
			p_y = []
			
			for i in df[p]:

				if i==None:
					p_x.append(None)
					p_y.append(None)
				
				else:
					p_x.append(float(i[0]))
					p_y.append(float(i[1]))
			
			return_df[f"{p}_x"] = p_x
			return_df[f"{p}_y"] = p_y

		for p in [col for col in df.columns if type(col)!=int]:
			
			return_df[p] = df[p]

		return return_df

	def create_frame_labels_df(self):

		self.df_frame_labels = pd.DataFrame(columns=["squat", "jump_peak", "landing"], index=range(1, self.total_frames+1), data=0)

	def convert_frame_labels_input(self, input_str):

		input_list = re.findall("[0-9]+", input_str)

		for i, frame in enumerate(input_list):
			input_list[i] = int(frame)

		input_list = sorted(input_list)

		return input_list

	def correct_frame_labels_input(self, input_list):

		if (0 <= min(input_list) <= self.total_frames) and (0 <= max(input_list) <= self.total_frames):
			return True
		else:
			return False

	def confirm_frames_input(self, frame_col, frames_input):

		if frames_input[0]==0:
			logging.debug(f"\tNo frames confirmed in '{frame_col}' column.")

		else:
			for frame in frames_input:
				self.df_frame_labels.loc[frame, frame_col] = 1
				logging.debug(f"\tFrame {frame} confirmed in '{frame_col}' column.")

	def create_dataframe_path(self, df_name):
		vid_path, vid_name_ext = os.path.split(self.video_path)
		vid_name, vid_ext = os.path.splitext(vid_name_ext)

		df_path = os.path.join(vid_path, vid_name)
		
		if not os.path.exists(df_path):
			os.makedirs(df_path)
			
		df_path = os.path.join(df_path, f"{vid_name}_{df_name}.csv")

		return df_path

	def save_data(self):
		# self.df_vid_points.to_csv(self.create_dataframe_path("FEATURES_BODY_POINTS"), index_label="frames")
		# self.df_vid_points_norm.to_csv(self.create_dataframe_path("FEATURES_BODY_POINTS_NORMALIZED"), index_label="frames")
		self.df_frame_labels.to_csv(self.create_dataframe_path("FRAME_LABELS"), index_label="frames")

	def create_zip_path(self):
		vid_path, vid_name_ext = os.path.split(self.video_path)
		vid_name, vid_ext = os.path.splitext(vid_name_ext)

		zip_dir = os.path.join(vid_path, vid_name)
		zip_path = os.path.join(vid_path, f"{vid_name}.zip")
		
		self.zip_path = zip_path

		return zip_dir, zip_path

	def zip_data(self):

		zip_dir, zip_path = self.create_zip_path()

		with zipfile.ZipFile(zip_path, "w") as myzip:
			for root, dirs, files in os.walk(zip_dir):
				for file in files:
					myzip.write(filename=os.path.join(zip_dir, file), arcname=file)