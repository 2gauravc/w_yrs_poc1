import numpy as np
import pandas as pd
import cv2
import os
import re
import zipfile
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import ndimage

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

				#cv2.imwrite(self.create_frame_path(total_frames), frame)

				# Exiting if "Q" key is pressed on the keyboard.
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		# Releasing the VideoCapture object.
		cap.release()
		
		self.total_frames = total_frames

	def new_final_vid_name(self, org_vid_path):
		vid_path, vid_name_ext = os.path.split(org_vid_path)
		vid_name, vid_ext = os.path.splitext(vid_name_ext)

		new_vid_name_ext = vid_name+"_ANALYSIS_VIDEO"+".mp4"
		new_vid_path = os.path.join(vid_path, new_vid_name_ext)

		return new_vid_path
	
	
	
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
		print("DataFrame saved to: {}".format(self.create_dataframe_path("FRAME_LABELS")))
    
	def create_frame_labels_df(self):
		self.df_frame_labels = pd.DataFrame(columns=["squat", "jump_peak", "landing"], index=range(1, self.total_frames+1), data=0)


	def analyse_vjump_video(self):
	
		logging.debug(f"Video analysis started.")

		# Creating a VideoCapture object.
		cap = cv2.VideoCapture(self.video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		print ("Video fps: {}".format(str(fps)))

		# Defining the codec and creating a VideoWriter object to save the output video at the same location.
	
		frame_counter = 1
		out = None

		while(cap.isOpened()):

			# Grabbing each individual frame, frame-by-frame.
			ret, frame = cap.read()
			if ret==True:
				rotated_frame = ndimage.rotate(frame, 270)
				cv2.putText(rotated_frame, "Frame Number: {}".format(frame_counter), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)
				# Saving frame to output video using the VideoWriter object defined above.
				if out is None:
					fourcc = cv2.VideoWriter_fourcc(*'MP4V')
					self.analysed_video_path = self.new_final_vid_name(self.video_path)
					(h, w) = rotated_frame.shape[:2]
					out = cv2.VideoWriter(self.analysed_video_path, fourcc, fps, (w, h))
 

				out.write(rotated_frame)

				frame_counter += 1

				# Exiting if "Q" key is pressed on the keyboard.
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		# Releasing the VideoCapture and VideoWriter objects, and closing the displayed frame.
		cap.release()
		out.release()
		
		print(f"Analysis video saved at {self.analysed_video_path}\n")
		logging.debug(f"Video analysis finished.")
		logging.debug(f"Analysis video saved at '{self.analysed_video_path}'.")
		
		#self.analysed_report_path = self.new_final_report_name(self.video_path)
	
		#print(f"Analysis report saved at {self.analysed_report_path}\n")
