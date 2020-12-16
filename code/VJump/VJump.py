import numpy as np
import pandas as pd
import cv2
import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns

import fastai
from fastai.vision.all import *
import pathlib
from pathlib import Path

class VJump():

	def __init__(self, video_path, orientation):

		self.video_path = video_path
		self.orientation = orientation

	def count_frames(self):

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

				# Exiting if "Q" key is pressed on the keyboard.
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		# Releasing the VideoCapture object.
		cap.release()
		
		self.total_frames = total_frames
		
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
				rotated_frame=ndimage.rotate(frame,270)
				cv2.imwrite(self.create_frame_path(total_frames), rotated_frame)

				# Exiting if "Q" key is pressed on the keyboard.
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		# Releasing the VideoCapture object.
		cap.release()
		
		self.total_frames = total_frames

	def create_frame_path(self, frame_counter):
		vid_path, vid_name_ext = os.path.split(self.video_path)
		vid_name, vid_ext = os.path.splitext(vid_name_ext)

		frame_path = os.path.join(vid_path, vid_name)
		
		if not os.path.exists(frame_path):
			os.makedirs(frame_path)
			
		frame_path = os.path.join(frame_path, f"{vid_name}_FRAME_{frame_counter}.png")
		return frame_path
	
	def fastai(self, model_path):
	    pathlib.WindowsPath = pathlib.PosixPath
		#print ("fastai: version {}".format(fastai.__version__))
	    model_path = Path(model_path)
	    learn_inf = load_learner(model_path)
		#print(learn_inf.dls.vocab)
	    print("Model Loaded")
	    self.fastai = learn_inf
	
	def import_body_net(self):

		# Source for code: https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

		# Specify the paths for the 2 files
		protoFile = "../inputs_outputs/models/pose_deploy.prototxt"
		weightsFile = "../inputs_outputs/models/pose_iter_584000.caffemodel"

		# Read the network into Memory
		self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
	
	def estimate_key_pose(self):
	    #Load the image from tmp/
	    temp_dir = '../inputs_outputs/IMG_3768'
	    g =  get_image_files(temp_dir)
	    
	    #Predict the Class for each image
	    for img in g:
	        lbl = self.fastai.predict(img)[0]
	        print("Image {}; Predicted Label {}".format(img, lbl))
	
	def find_skeleton(self, frame, frame_counter):

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
				cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(frame_copy, f"{i}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
				cv2.putText(frame_copy, f"frame = {frame_counter}", (10, self.frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)

				# Add the point to the list if the probability is greater than the threshold
				points.append((int(x), int(y)))

			else:
				points.append(None)
		
		return frame_copy, points
	
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
				skeleton_frame, skeleton_frame_points = self.find_skeleton(frame, frame_counter)

				# Saving frame to output video using the VideoWriter object defined above.
				# out.write(skeleton_frame)

				# Displaying the frame with age detected.
				# cv2.imshow("Output Video", skeleton_frame)

				# Saving frame output skeleton point coordinates as a new row in above defined dataframe.
				df_vid_points[frame_counter] = skeleton_frame_points
				frame_counter += 1

				# Exiting if "Q" key is pressed on the keyboard.
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			else:
				break

		# Releasing the VideoCapture and VideoWriter objects, and closing the displayed frame.
		cap.release()
		# out.release()
		# cv2.destroyAllWindows()
		# print(f"Saved to {new_my_video}")
		
		df_vid_points = df_vid_points.T.copy()
		
		critical_points = [0, 1, 8]
		if self.orientation == "right":
			critical_points.extend([2, 3, 4, 9, 10, 11])
		elif self.orientation == "left":
			critical_points.extend([5, 6, 7, 12, 13, 14])
			
		df_vid_points = df_vid_points[critical_points]
		
		self.df_vid_points = df_vid_points
		
		self.df_vid_points_imputed = self.df_vid_points.copy()
		
	def identify_critical_frames(self):

		forehead_y_movement = [(self.frame_height - x[1]) if x!=None else None for x in self.df_vid_points[0]]

		jump_peak_indexes = []
		max_forehead_y_movement = np.nanmax(np.array(forehead_y_movement, dtype=np.float64)).astype(int)

		for i, y in enumerate(forehead_y_movement):
			if y == max_forehead_y_movement:
				jump_peak_indexes.append(i+1)

		self.jump_peak_frame = int(np.median(jump_peak_indexes))

		jump_squat_indexes = []
		min_squat_forehead_y_movement = np.nanmin(np.array(forehead_y_movement[:self.jump_peak_frame], dtype=np.float64)).astype(int)

		for i, y in enumerate(forehead_y_movement[:self.jump_peak_frame]):
			if y == min_squat_forehead_y_movement:
				jump_squat_indexes.append(i+1)

		self.jump_squat_frame = int(np.median(jump_squat_indexes))

		jump_land_indexes = []
		min_land_forehead_y_movement = np.nanmin(np.array(forehead_y_movement[self.jump_peak_frame:], dtype=np.float64)).astype(int)

		for i, y in enumerate(forehead_y_movement[self.jump_peak_frame:]):
			if y == min_land_forehead_y_movement:
				jump_land_indexes.append(i+1+self.jump_peak_frame)

		self.jump_land_frame = int(np.median(jump_land_indexes))
	
	def import_vjump_criteria(self):
		
		try:
			df_vjump_criteria = pd.read_csv("../inputs_outputs/vjump_criteria.csv")
			df_vjump_criteria["condition"].fillna(value="", inplace=True)
			logging.debug(f"Success criteria CSV file FOUND.")
		
		except:
			print("Error importing jump criteria file. Reverting to default criteria.")
			logging.debug(f"Success criteria CSV file NOT FOUND. Reverting to default criteria.")
			
			vjump_default_criteria_dict = {'frame': {0: 'SQUAT POSTURE ANGLES',
													 1: 'SQUAT POSTURE ANGLES',
													 2: 'SQUAT POSTURE ANGLES',
													 3: 'SQUAT POSTURE ANGLES',
													 4: 'JUMP PEAK POSTURE ANGLES',
													 5: 'JUMP PEAK POSTURE ANGLES',
													 6: 'JUMP PEAK POSTURE ANGLES',
													 7: 'LANDING POSTURE ANGLES',
													 8: 'LANDING POSTURE ANGLES'},
										   'criteria_description': {0: 'Elbow angle',
																	1: 'Knee angle',
																	2: 'Shoulder angle',
																	3: 'Hip angle',
																	4: 'Elbow angle',
																	5: 'Knee angle',
																	6: 'Torso leaning angle',
																	7: 'Knee angle',
																	8: 'Torso leaning angle'},
										   'condition': {0: '>=',
														 1: '[ , ]',
														 2: '',
														 3: '',
														 4: '>=',
														 5: '>=',
														 6: '<=',
														 7: '<=',
														 8: '<='},
										   'min_value': {0: 160.0,
														 1: 75.0,
														 2: None,
														 3: None,
														 4: 150.0,
														 5: 150.0,
														 6: None,
														 7: None,
														 8: None},
										   'max_value': {0: None,
														 1: 105.0,
														 2: None,
														 3: None,
														 4: None,
														 5: None,
														 6: 20.0,
														 7: 150.0,
														 8: 10.0}
										  }
			
			df_vjump_criteria = pd.DataFrame(vjump_default_criteria_dict)
			df_vjump_criteria.to_csv("../inputs_outputs/vjump_criteria.csv", index=False)
					
		self.df_vjump_criteria = df_vjump_criteria
	
	def need_to_impute_critical_frames(self):
		
		critical_points = []
		
		if self.orientation == "right":
			critical_points = [2, 3, 4, 9, 10, 11]
		elif self.orientation == "left":
			critical_points = [5, 6, 7, 12, 13, 14]
			
		need_to_impute = False
		
		logging.debug(f"Starting adjustment of critical frames to nearby frames.")

		jump_squat_frame_adjusters = [0, -1, 1, -2, 2]
		temp_frame = self.jump_squat_frame
		
		for adj in jump_squat_frame_adjusters:
			if self.df_vid_points_imputed.loc[temp_frame + adj, critical_points].isna().sum() == 0:
				self.jump_squat_frame = temp_frame + adj
				# print(f"NEW JUMP SQUAT FRAME = {self.jump_squat_frame}")
				logging.debug(f"ADJUSTED Squat posture: frame {self.jump_squat_frame}")
				break
			elif adj == jump_squat_frame_adjusters[-1]:
				need_to_impute = True
				# print(f"SQUAT FRAME NEED TO IMPUTE = {need_to_impute}")
				logging.debug(f"UNADJUSTED Squat posture: frame {self.jump_squat_frame}")
				logging.debug(f"Nearby frames also have missing data. NEED TO IMPUTE squat posture frame.")
				return need_to_impute
			
		jump_peak_frame_adjusters = [0, -1, 1, -2, 2]
		temp_frame = self.jump_peak_frame
		
		for adj in jump_peak_frame_adjusters:
			if self.df_vid_points_imputed.loc[temp_frame + adj, critical_points].isna().sum() == 0:
				self.jump_peak_frame = temp_frame + adj
				# print(f"NEW JUMP PEAK FRAME = {self.jump_peak_frame}")
				logging.debug(f"ADJUSTED Jump Peak posture: frame {self.jump_peak_frame}")
				break
			elif adj == jump_peak_frame_adjusters[-1]:
				need_to_impute = True
				# print(f"PEAK FRAME NEED TO IMPUTE = {need_to_impute}")
				logging.debug(f"UNADJUSTED Jump Peak posture: frame {self.jump_peak_frame}")
				logging.debug(f"Nearby frames also have missing data. NEED TO IMPUTE jump peak posture frame.")
				return need_to_impute
			
		jump_land_frame_adjusters = [0, -1, 1, -2, 2]
		temp_frame = self.jump_land_frame
		
		for adj in jump_land_frame_adjusters:
			if self.df_vid_points_imputed.loc[temp_frame + adj, critical_points].isna().sum() == 0:
				self.jump_land_frame = temp_frame + adj
				# print(f"NEW JUMP LAND FRAME = {self.jump_land_frame}")
				logging.debug(f"ADJUSTED Landing posture: frame {self.jump_land_frame}")
				break
			elif adj == jump_land_frame_adjusters[-1]:
				need_to_impute = True
				# print(f"LAND FRAME NEED TO IMPUTE = {need_to_impute}")
				logging.debug(f"UNADJUSTED Landing posture: frame {self.jump_land_frame}")
				logging.debug(f"Nearby frames also have missing data. NEED TO IMPUTE landing posture frame.")
				return need_to_impute
		
		logging.debug(f"Successfully adjusted critical frames to nearby frames to avoid data imputation.")

		return need_to_impute
	
	def impute_missing_body_points(self):
		
		from VJump import Body_Points_Imputation

		for i in list(self.df_vid_points_imputed.columns):
			self.df_vid_points_imputed[i] = Body_Points_Imputation.imputing_missing_points_in_series(self.df_vid_points_imputed[i])
			# self.df_vid_points_imputed[i] = imputing_missing_points_in_series(self.df_vid_points_imputed[i])
			
	def calculate_angle_between_three_points(self, point_1, point_2, point_3):
	
		x_1, y_1 = point_1
		x_2, y_2 = point_2
		x_3, y_3 = point_3

		side_1_2 = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
		side_2_3 = np.sqrt((x_3 - x_2)**2 + (y_3 - y_2)**2)
		side_3_1 = np.sqrt((x_1 - x_3)**2 + (y_1 - y_3)**2)

		cos_theta_rad = ((side_1_2**2 + side_2_3**2 - side_3_1**2) / (2 * side_1_2 * side_2_3))

		theta_rad = np.arccos(cos_theta_rad)

		theta_deg = np.rad2deg(theta_rad)

		return theta_deg
	
	def jump_quality_at_squat_frame(self, frame, frame_counter, points_row_original, points_row_imputed):
	
		frame_copy = np.copy(frame)

		elbow_angle_points = []
		knee_angle_points = []
		shoulder_angle_points = []
		hip_angle_points = []

		if self.orientation == "right":
			elbow_angle_points = [2, 3, 4]
			knee_angle_points = [9, 10, 11]
			shoulder_angle_points = [3, 2, 9]
			hip_angle_points = [2, 9, 10]
		elif self.orientation == "left":
			elbow_angle_points = [5, 6, 7]
			knee_angle_points = [12, 13, 14]
			shoulder_angle_points = [6, 5, 12]
			hip_angle_points = [5, 12, 13]

		logging.debug(f"\t- Calculating body angles.")

		elbow_angle = self.calculate_angle_between_three_points(points_row_imputed[elbow_angle_points[0]],
																points_row_imputed[elbow_angle_points[1]],
																points_row_imputed[elbow_angle_points[2]]
															   )

		knee_angle = self.calculate_angle_between_three_points(points_row_imputed[knee_angle_points[0]],
															   points_row_imputed[knee_angle_points[1]],
															   points_row_imputed[knee_angle_points[2]]
															  )

		shoulder_angle = self.calculate_angle_between_three_points(points_row_imputed[shoulder_angle_points[0]],
																   points_row_imputed[shoulder_angle_points[1]],
																   points_row_imputed[shoulder_angle_points[2]]
																  )

		hip_angle = self.calculate_angle_between_three_points(points_row_imputed[hip_angle_points[0]],
															  points_row_imputed[hip_angle_points[1]],
															  points_row_imputed[hip_angle_points[2]]
															 )

		df_squat_criteria = self.df_vjump_criteria[self.df_vjump_criteria["frame"]=="SQUAT POSTURE ANGLES"]
		
		elbow_angle_pass = False
		knee_angle_pass = False
		shoulder_angle_pass = False
		hip_angle_pass = False
		
		logging.debug(f"\t- Judging elbow angle.")
		elbow_angle_condition = df_squat_criteria[df_squat_criteria["criteria_description"]=="Elbow angle"]["condition"].tolist()[0]
		elbow_min_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Elbow angle"]["min_value"].tolist()[0]
		elbow_max_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Elbow angle"]["max_value"].tolist()[0]
		elbow_angle_criteria_text = ""
		
		if elbow_angle_condition == "":
			elbow_angle_pass = True
		elif elbow_angle_condition == ">=":
			elbow_angle_criteria_text = f"[correct range: >={elbow_min_angle} deg]"
			if elbow_angle >= elbow_min_angle:
				elbow_angle_pass = True
		elif elbow_angle_condition == "<=":
			elbow_angle_criteria_text = f"[correct range: <={elbow_max_angle} deg]"
			if elbow_angle <= elbow_max_angle:
				elbow_angle_pass = True
		elif elbow_angle_condition == "[ , ]":
			elbow_angle_criteria_text = f"[correct range: {elbow_min_angle}-{elbow_max_angle} deg]"
			if elbow_min_angle <= elbow_angle <= elbow_max_angle:
				elbow_angle_pass = True
		logging.debug(f"\t- Elbow: {round(elbow_angle, 2)} deg\t{elbow_angle_criteria_text}")
		
		logging.debug(f"\t- Judging knee angle.")
		knee_angle_condition = df_squat_criteria[df_squat_criteria["criteria_description"]=="Knee angle"]["condition"].tolist()[0]
		knee_min_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Knee angle"]["min_value"].tolist()[0]
		knee_max_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Knee angle"]["max_value"].tolist()[0]
		knee_angle_criteria_text = ""
		
		if knee_angle_condition == "":
			knee_angle_pass = True
		elif knee_angle_condition == ">=":
			knee_angle_criteria_text = f"[correct range: >={knee_min_angle} deg]"
			if knee_angle >= knee_min_angle:
				knee_angle_pass = True
		elif knee_angle_condition == "<=":
			knee_angle_criteria_text = f"[correct range: <={knee_max_angle} deg]"
			if knee_angle <= knee_max_angle:
				knee_angle_pass = True
		elif knee_angle_condition == "[ , ]":
			knee_angle_criteria_text = f"[correct range: {knee_min_angle}-{knee_max_angle} deg]"
			if knee_min_angle <= knee_angle <= knee_max_angle:
				knee_angle_pass = True
		logging.debug(f"\t- Knee: {round(knee_angle, 2)} deg\t{knee_angle_criteria_text}")
		
		logging.debug(f"\t- Judging shoulder angle.")
		shoulder_angle_condition = df_squat_criteria[df_squat_criteria["criteria_description"]=="Shoulder angle"]["condition"].tolist()[0]
		shoulder_min_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Shoulder angle"]["min_value"].tolist()[0]
		shoulder_max_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Shoulder angle"]["max_value"].tolist()[0]
		shoulder_angle_criteria_text = ""
		
		if shoulder_angle_condition == "":
			shoulder_angle_pass = True
		elif shoulder_angle_condition == ">=":
			shoulder_angle_criteria_text = f"[correct range: >={shoulder_min_angle} deg]"
			if shoulder_angle >= shoulder_min_angle:
				shoulder_angle_pass = True
		elif shoulder_angle_condition == "<=":
			shoulder_angle_criteria_text = f"[correct range: <={shoulder_max_angle} deg]"
			if shoulder_angle <= shoulder_max_angle:
				shoulder_angle_pass = True
		elif shoulder_angle_condition == "[ , ]":
			shoulder_angle_criteria_text = f"[correct range: {shoulder_min_angle}-{shoulder_max_angle} deg]"
			if shoulder_min_angle <= shoulder_angle <= shoulder_max_angle:
				shoulder_angle_pass = True
		logging.debug(f"\t- Shldr: {round(shoulder_angle, 2)} deg\t{shoulder_angle_criteria_text}")
		
		logging.debug(f"\t- Judging hip angle.")
		hip_angle_condition = df_squat_criteria[df_squat_criteria["criteria_description"]=="Hip angle"]["condition"].tolist()[0]
		hip_min_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Hip angle"]["min_value"].tolist()[0]
		hip_max_angle = df_squat_criteria[df_squat_criteria["criteria_description"]=="Hip angle"]["max_value"].tolist()[0]
		hip_angle_criteria_text = ""
		
		if hip_angle_condition == "":
			hip_angle_pass = True
		elif hip_angle_condition == ">=":
			hip_angle_criteria_text = f"[correct range: >={hip_min_angle} deg]"
			if hip_angle >= hip_min_angle:
				hip_angle_pass = True
		elif hip_angle_condition == "<=":
			hip_angle_criteria_text = f"[correct range: <={hip_max_angle} deg]"
			if hip_angle <= hip_max_angle:
				hip_angle_pass = True
		elif hip_angle_condition == "[ , ]":
			hip_angle_criteria_text = f"[correct range: {hip_min_angle}-{hip_max_angle} deg]"
			if hip_min_angle <= hip_angle <= hip_max_angle:
				hip_angle_pass = True
		logging.debug(f"\t- Hip: {round(hip_angle, 2)} deg\t{hip_angle_criteria_text}")
		
		logging.debug(f"\t- Drawing red lines indicating any incorrect squat posture.")

		if elbow_angle_pass == False:
			for i, p in enumerate(elbow_angle_points[:-1]):
				cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[elbow_angle_points[i+1]], (0, 0, 255), thickness=2)
		
		if knee_angle_pass == False:
			for i, p in enumerate(knee_angle_points[:-1]):
				cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[knee_angle_points[i+1]], (0, 0, 255), thickness=2)
		
		if shoulder_angle_pass == False:
			for i, p in enumerate(shoulder_angle_points[:-1]):
				cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[shoulder_angle_points[i+1]], (0, 0, 255), thickness=2)
		
		if hip_angle_pass == False:
			for i, p in enumerate(hip_angle_points[:-1]):
				cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[hip_angle_points[i+1]], (0, 0, 255), thickness=2)

		logging.debug(f"\t- Drawing all body points and frame number.")
		for i in points_row_imputed.index:
			if points_row_imputed[i] != None:
				if points_row_original[i] != None:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				else:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
			cv2.putText(frame_copy, f"frame = {frame_counter}", (10, self.frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)

		frame_text = "Squat posture : FAIL"
		if elbow_angle_pass and knee_angle_pass and shoulder_angle_pass and hip_angle_pass:
			frame_text = "Squat posture : PASS"

		logging.debug(f"\t- {frame_text}")

		report_text = f'''
		{frame_text} (frame {frame_counter} of {self.total_frames})
		Body angles:
		\tElbow:\t{round(elbow_angle, 2)} deg\t{elbow_angle_criteria_text}
		\tKnee:\t{round(knee_angle, 2)} deg\t{knee_angle_criteria_text}
		\tShldr:\t{round(shoulder_angle, 2)} deg\t{shoulder_angle_criteria_text}
		\tHip:\t{round(knee_angle, 2)} deg\t{hip_angle_criteria_text}
		'''

		return frame_copy, frame_text, report_text
	
	def jump_quality_at_peak_frame(self, frame, frame_counter, points_row_original, points_row_imputed):
	
		frame_copy = np.copy(frame)

		elbow_angle_points = []
		knee_angle_points = []
		torso_leaning_angle_points = []

		if self.orientation == "right":
			elbow_angle_points = [2, 3, 4]
			knee_angle_points = [9, 10, 11]
			torso_leaning_angle_points = [2, 2, 9]
		elif self.orientation == "left":
			elbow_angle_points = [5, 6, 7]
			knee_angle_points = [12, 13, 14]
			torso_leaning_angle_points = [5, 5, 12]
		
		logging.debug(f"\t- Calculating body angles.")

		elbow_angle = self.calculate_angle_between_three_points(points_row_imputed[elbow_angle_points[0]],
																points_row_imputed[elbow_angle_points[1]],
																points_row_imputed[elbow_angle_points[2]]
															   )

		knee_angle = self.calculate_angle_between_three_points(points_row_imputed[knee_angle_points[0]],
															   points_row_imputed[knee_angle_points[1]],
															   points_row_imputed[knee_angle_points[2]]
															  )

		torso_leaning_angle = self.calculate_angle_between_three_points((points_row_imputed[torso_leaning_angle_points[0]][0], points_row_imputed[torso_leaning_angle_points[0]][1] + 10),
																		points_row_imputed[torso_leaning_angle_points[1]],
																		points_row_imputed[torso_leaning_angle_points[2]]
																	   )
		
		df_peak_criteria = self.df_vjump_criteria[self.df_vjump_criteria["frame"]=="JUMP PEAK POSTURE ANGLES"]
		
		elbow_angle_pass = False
		knee_angle_pass = False
		torso_leaning_angle_pass = False
		
		logging.debug(f"\t- Judging elbow angle.")
		elbow_angle_condition = df_peak_criteria[df_peak_criteria["criteria_description"]=="Elbow angle"]["condition"].tolist()[0]
		elbow_min_angle = df_peak_criteria[df_peak_criteria["criteria_description"]=="Elbow angle"]["min_value"].tolist()[0]
		elbow_max_angle = df_peak_criteria[df_peak_criteria["criteria_description"]=="Elbow angle"]["max_value"].tolist()[0]
		elbow_angle_criteria_text = ""
		
		if elbow_angle_condition == "":
			elbow_angle_pass = True
		elif elbow_angle_condition == ">=":
			elbow_angle_criteria_text = f"[correct range: >={elbow_min_angle} deg]"
			if elbow_angle >= elbow_min_angle:
				elbow_angle_pass = True
		elif elbow_angle_condition == "<=":
			elbow_angle_criteria_text = f"[correct range: <={elbow_max_angle} deg]"
			if elbow_angle <= elbow_max_angle:
				elbow_angle_pass = True
		elif elbow_angle_condition == "[ , ]":
			elbow_angle_criteria_text = f"[correct range: {elbow_min_angle}-{elbow_max_angle} deg]"
			if elbow_min_angle <= elbow_angle <= elbow_max_angle:
				elbow_angle_pass = True
		logging.debug(f"\t- Elbow: {round(elbow_angle, 2)} deg\t{elbow_angle_criteria_text}")
		
		logging.debug(f"\t- Judging knee angle.")
		knee_angle_condition = df_peak_criteria[df_peak_criteria["criteria_description"]=="Knee angle"]["condition"].tolist()[0]
		knee_min_angle = df_peak_criteria[df_peak_criteria["criteria_description"]=="Knee angle"]["min_value"].tolist()[0]
		knee_max_angle = df_peak_criteria[df_peak_criteria["criteria_description"]=="Knee angle"]["max_value"].tolist()[0]
		knee_angle_criteria_text = ""
		
		if knee_angle_condition == "":
			knee_angle_pass = True
		elif knee_angle_condition == ">=":
			knee_angle_criteria_text = f"[correct range: >={knee_min_angle} deg]"
			if knee_angle >= knee_min_angle:
				knee_angle_pass = True
		elif knee_angle_condition == "<=":
			knee_angle_criteria_text = f"[correct range: <={knee_max_angle} deg]"
			if knee_angle <= knee_max_angle:
				knee_angle_pass = True
		elif knee_angle_condition == "[ , ]":
			knee_angle_criteria_text = f"[correct range: {knee_min_angle}-{knee_max_angle} deg]"
			if knee_min_angle <= knee_angle <= knee_max_angle:
				knee_angle_pass = True
		logging.debug(f"\t- Knee: {round(knee_angle, 2)} deg\t{knee_angle_criteria_text}")
		
		logging.debug(f"\t- Judging torso leaning angle.")
		torso_leaning_angle_condition = df_peak_criteria[df_peak_criteria["criteria_description"]=="Torso leaning angle"]["condition"].tolist()[0]
		torso_leaning_min_angle = df_peak_criteria[df_peak_criteria["criteria_description"]=="Torso leaning angle"]["min_value"].tolist()[0]
		torso_leaning_max_angle = df_peak_criteria[df_peak_criteria["criteria_description"]=="Torso leaning angle"]["max_value"].tolist()[0]
		torso_leaning_angle_criteria_text = ""
		
		if torso_leaning_angle_condition == "":
			torso_leaning_angle_pass = True
		elif torso_leaning_angle_condition == ">=":
			torso_leaning_angle_criteria_text = f"[correct range: >={torso_leaning_min_angle} deg]"
			if torso_leaning_angle >= torso_leaning_min_angle:
				torso_leaning_angle_pass = True
		elif torso_leaning_angle_condition == "<=":
			torso_leaning_angle_criteria_text = f"[correct range: <={torso_leaning_max_angle} deg]"
			if torso_leaning_angle <= torso_leaning_max_angle:
				torso_leaning_angle_pass = True
		elif torso_leaning_angle_condition == "[ , ]":
			torso_leaning_angle_criteria_text = f"[correct range: {torso_leaning_min_angle}-{torso_leaning_max_angle} deg]"
			if torso_leaning_min_angle <= torso_leaning_angle <= torso_leaning_max_angle:
				torso_leaning_angle_pass = True
		logging.debug(f"\t- Torso: {round(torso_leaning_angle, 2)} deg\t{torso_leaning_angle_criteria_text}")
		
		logging.debug(f"\t- Drawing red lines indicating any incorrect jump peak posture.")

		if elbow_angle_pass == False:
			for i, p in enumerate(elbow_angle_points[:-1]):
				cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[elbow_angle_points[i+1]], (0, 0, 255), thickness=2)
		
		if knee_angle_pass == False:
			for i, p in enumerate(knee_angle_points[:-1]):
				cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[knee_angle_points[i+1]], (0, 0, 255), thickness=2)
		
		if torso_leaning_angle_pass == False:
			cv2.line(frame_copy, points_row_imputed[torso_leaning_angle_points[1]], points_row_imputed[torso_leaning_angle_points[2]], (0, 0, 255), thickness=2)
			cv2.line(frame_copy, points_row_imputed[torso_leaning_angle_points[2]], (points_row_imputed[torso_leaning_angle_points[2]][0], points_row_imputed[torso_leaning_angle_points[2]][1] - 100), (0, 0, 255), thickness=2)
			
		logging.debug(f"\t- Drawing all body points and frame number.")
		for i in points_row_imputed.index:
			if points_row_imputed[i] != None:
				if points_row_original[i] != None:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				else:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
			cv2.putText(frame_copy, f"frame = {frame_counter}", (10, self.frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)

		frame_text = "Jump Peak posture : FAIL"
		if elbow_angle_pass and knee_angle_pass and torso_leaning_angle_pass:
			frame_text = "Jump Peak posture : PASS"

		logging.debug(f"\t- {frame_text}")
		
		report_text = f'''
		{frame_text} (frame {frame_counter} of {self.total_frames})
		Body angles:
		\tElbow:\t{round(elbow_angle, 2)} deg\t{elbow_angle_criteria_text}
		\tKnee:\t{round(knee_angle, 2)} deg\t{knee_angle_criteria_text}
		\tTorso:\t{round(torso_leaning_angle, 2)} deg\t{torso_leaning_angle_criteria_text}
		'''

		return frame_copy, frame_text, report_text
	
	def find_upper_body_length(self):
	
		upper_body_length = 0

		for i in self.df_vid_points_imputed.index:

			if ((self.df_vid_points_imputed.loc[i, 1] != None) and (self.df_vid_points_imputed.loc[i, 8] != None)):

				upper_chest_x, upper_chest_y = self.df_vid_points_imputed.loc[i, 1]
				navel_x, navel_y = self.df_vid_points_imputed.loc[i, 8]

				upper_body_length = int(round(np.sqrt((upper_chest_x - navel_x)**2 + (upper_chest_y - navel_y)**2)))

				break

		return upper_body_length
	
	def calculate_distance_from_squat_to_land(self):
	
		logging.debug(f"\t- Finding upper body length in pixels.")
		upper_body_length = self.find_upper_body_length()
		logging.debug(f"\t- Found upper body length = {upper_body_length} pixels.")

		ankle_point = 0
		
		if self.orientation == "right":
			ankle_point = 11
		elif self.orientation == "left":
			ankle_point = 14

		squat_ankle_x, squat_ankle_y = self.df_vid_points_imputed.loc[self.jump_squat_frame, ankle_point]
		land_ankle_x, land_ankle_y = self.df_vid_points_imputed.loc[self.jump_land_frame, ankle_point]

		landing_distance = np.sqrt((land_ankle_x - squat_ankle_x)**2 + (land_ankle_y - squat_ankle_y)**2)
		logging.debug(f"\t- Found landing distance = {landing_distance} pixels.")

		return landing_distance / upper_body_length
	
	def jump_quality_at_land_frame(self, frame, frame_counter, points_row_original, points_row_imputed):
	
		frame_copy = np.copy(frame)

		knee_angle_points = []
		torso_leaning_angle_points = []

		if self.orientation == "right":
			knee_angle_points = [9, 10, 11]
			torso_leaning_angle_points = [2, 2, 9]
		elif self.orientation == "left":
			knee_angle_points = [12, 13, 14]
			torso_leaning_angle_points = [5, 5, 12]
		
		logging.debug(f"\t- Calculating body angles.")

		knee_angle = self.calculate_angle_between_three_points(points_row_imputed[knee_angle_points[0]],
															   points_row_imputed[knee_angle_points[1]],
															   points_row_imputed[knee_angle_points[2]]
															  )

		torso_leaning_angle = self.calculate_angle_between_three_points((points_row_imputed[torso_leaning_angle_points[0]][0], points_row_imputed[torso_leaning_angle_points[0]][1] + 10),
																		points_row_imputed[torso_leaning_angle_points[1]],
																		points_row_imputed[torso_leaning_angle_points[2]]
																	   )
		
		df_land_criteria = self.df_vjump_criteria[self.df_vjump_criteria["frame"]=="LANDING POSTURE ANGLES"]
		
		knee_angle_pass = False
		torso_leaning_angle_pass = False
		land_distance_pass = False
		
		logging.debug(f"\t- Judging knee angle.")
		knee_angle_condition = df_land_criteria[df_land_criteria["criteria_description"]=="Knee angle"]["condition"].tolist()[0]
		knee_min_angle = df_land_criteria[df_land_criteria["criteria_description"]=="Knee angle"]["min_value"].tolist()[0]
		knee_max_angle = df_land_criteria[df_land_criteria["criteria_description"]=="Knee angle"]["max_value"].tolist()[0]
		knee_angle_criteria_text = ""
		
		if knee_angle_condition == "":
			knee_angle_pass = True
		elif knee_angle_condition == ">=":
			knee_angle_criteria_text = f"[correct range: >={knee_min_angle} deg]"
			if knee_angle >= knee_min_angle:
				knee_angle_pass = True
		elif knee_angle_condition == "<=":
			knee_angle_criteria_text = f"[correct range: <={knee_max_angle} deg]"
			if knee_angle <= knee_max_angle:
				knee_angle_pass = True
		elif knee_angle_condition == "[ , ]":
			knee_angle_criteria_text = f"[correct range: {knee_min_angle}-{knee_max_angle} deg]"
			if knee_min_angle <= knee_angle <= knee_max_angle:
				knee_angle_pass = True
		logging.debug(f"\t- Knee: {round(knee_angle, 2)} deg\t{knee_angle_criteria_text}")
		
		logging.debug(f"\t- Judging torso leaning angle.")
		torso_leaning_angle_condition = df_land_criteria[df_land_criteria["criteria_description"]=="Torso leaning angle"]["condition"].tolist()[0]
		torso_leaning_min_angle = df_land_criteria[df_land_criteria["criteria_description"]=="Torso leaning angle"]["min_value"].tolist()[0]
		torso_leaning_max_angle = df_land_criteria[df_land_criteria["criteria_description"]=="Torso leaning angle"]["max_value"].tolist()[0]
		torso_leaning_angle_criteria_text = ""
		
		if torso_leaning_angle_condition == "":
			torso_leaning_angle_pass = True
		elif torso_leaning_angle_condition == ">=":
			torso_leaning_angle_criteria_text = f"[correct range: >={torso_leaning_min_angle} deg]"
			if torso_leaning_angle >= torso_leaning_min_angle:
				torso_leaning_angle_pass = True
		elif torso_leaning_angle_condition == "<=":
			torso_leaning_angle_criteria_text = f"[correct range: <={torso_leaning_max_angle} deg]"
			if torso_leaning_angle <= torso_leaning_max_angle:
				torso_leaning_angle_pass = True
		elif torso_leaning_angle_condition == "[ , ]":
			torso_leaning_angle_criteria_text = f"[correct range: {torso_leaning_min_angle}-{torso_leaning_max_angle} deg]"
			if torso_leaning_min_angle <= torso_leaning_angle <= torso_leaning_max_angle:
				torso_leaning_angle_pass = True
		logging.debug(f"\t- Torso: {round(torso_leaning_angle, 2)} deg\t{torso_leaning_angle_criteria_text}")
		
		logging.debug(f"\t- Judging landing distance.")
		land_distance = self.calculate_distance_from_squat_to_land()
		if land_distance <= 1:
			land_distance_pass = True
		logging.debug(f"\t- Landing distance: {round(land_distance, 2)} X Upper Body length [correct range: <=1]")
		
		logging.debug(f"\t- Drawing red lines indicating any incorrect landing posture.")

		if knee_angle_pass == False:
			for i, p in enumerate(knee_angle_points[:-1]):
				cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[knee_angle_points[i+1]], (0, 0, 255), thickness=2)
		
		if torso_leaning_angle_pass == False:
			cv2.line(frame_copy, points_row_imputed[torso_leaning_angle_points[1]], points_row_imputed[torso_leaning_angle_points[2]], (0, 0, 255), thickness=2)
			cv2.line(frame_copy, points_row_imputed[torso_leaning_angle_points[2]], (points_row_imputed[torso_leaning_angle_points[2]][0], points_row_imputed[torso_leaning_angle_points[2]][1] - 100), (0, 0, 255), thickness=2)
		
		logging.debug(f"\t- Drawing all body points and frame number.")
		for i in points_row_imputed.index:
			if points_row_imputed[i] != None:
				if points_row_original[i] != None:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				else:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
			cv2.putText(frame_copy, f"frame = {frame_counter}", (10, self.frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
		
		frame_text = "Landing posture : FAIL"
		if knee_angle_pass and torso_leaning_angle_pass and land_distance_pass:
			frame_text = "Landing posture : PASS"

		logging.debug(f"\t- {frame_text}")
		
		report_text = f'''
		{frame_text} (frame {frame_counter} of {self.total_frames})
		Body angles:
		\tKnee:\t{round(knee_angle, 2)} deg\t{knee_angle_criteria_text}
		\tTorso:\t{round(torso_leaning_angle, 2)} deg\t{torso_leaning_angle_criteria_text}

		Landing distance: {round(land_distance, 2)} X Upper Body length [correct range: <=1]
		'''

		return frame_copy, frame_text, report_text
	
	def overlay_imputed_skeleton(self, frame, frame_counter, points_row_original, points_row_imputed):

		frame_copy = np.copy(frame)
		
		logging.debug(f"\t- Drawing all body points and frame number.")
		for i in points_row_imputed.index:
			if points_row_imputed[i] != None:
				if points_row_original[i] != None:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				else:
					cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
			cv2.putText(frame_copy, f"frame = {frame_counter}", (10, self.frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
		
		return frame_copy
	
	def overlay_frame_text(self, frame, squat_frame_text, peak_frame_text, land_frame_text):
	
		frame_copy = np.copy(frame)

		logging.debug(f"\t- Drawing frame text.")

		if "PASS" in squat_frame_text:
			cv2.putText(frame_copy, squat_frame_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, lineType=cv2.LINE_AA)
		elif "FAIL" in squat_frame_text:
			cv2.putText(frame_copy, squat_frame_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
		else:
			cv2.putText(frame_copy, "Squat posture : ANALYSING...", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)

		if "PASS" in peak_frame_text:
			cv2.putText(frame_copy, peak_frame_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, lineType=cv2.LINE_AA)
		elif "FAIL" in peak_frame_text:
			cv2.putText(frame_copy, peak_frame_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
		else:
			cv2.putText(frame_copy, "Jump Peak posture : ANALYSING...", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)

		if "PASS" in land_frame_text:
			cv2.putText(frame_copy, land_frame_text, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, lineType=cv2.LINE_AA)
		elif "FAIL" in land_frame_text:
			cv2.putText(frame_copy, land_frame_text, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
		else:
			cv2.putText(frame_copy, "Landing posture : ANALYSING...", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)

		return frame_copy
	
	# Defining a function to return the video filepath with a new filename.
	# If INPUT filepath is "my_folder1/my_folder2/my_video.mp4", OUTPUT filepath will be "my_folder1/my_folder2/my_video_WITH_AGE.mp4"

	def new_final_vid_name(self, org_vid_path):
		vid_path, vid_name_ext = os.path.split(org_vid_path)
		vid_name, vid_ext = os.path.splitext(vid_name_ext)

		new_vid_name_ext = vid_name+"_ANALYSIS_VIDEO"+".mp4"
		new_vid_path = os.path.join(vid_path, new_vid_name_ext)

		return new_vid_path
	
	def new_final_report_name(self, org_vid_path):
		vid_path, vid_name_ext = os.path.split(org_vid_path)
		vid_name, vid_ext = os.path.splitext(vid_name_ext)

		report_name_ext = vid_name+"_ANALYSIS_REPORT"+".txt"
		report_path = os.path.join(vid_path, report_name_ext)

		return report_path
	
	def analyse_vjump_video(self):
	
		logging.debug(f"Video analysis started.")

		# Creating a VideoCapture object.
		cap = cv2.VideoCapture(self.video_path)

		# Defining the codec and creating a VideoWriter object to save the output video at the same location.
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		self.analysed_video_path = self.new_final_vid_name(self.video_path)
		out = cv2.VideoWriter(self.analysed_video_path, fourcc, 18, (self.frame_width, self.frame_height))
	
		frame_counter = 1
		squat_frame_text = ""
		peak_frame_text = ""
		land_frame_text = ""

		final_report_text = f'''VERTICAL JUMP REPORT
		Video Path : {self.video_path}
		'''

		while(cap.isOpened()):

			# Grabbing each individual frame, frame-by-frame.
			ret, frame = cap.read()

			if ret==True:

				if frame_counter == self.jump_squat_frame:
					logging.debug(f"Analysing video - frame {frame_counter} of {self.total_frames} - SQUAT posture frame.")
					skeleton_frame, squat_frame_text, report_text = self.jump_quality_at_squat_frame(frame, frame_counter, self.df_vid_points.loc[frame_counter], self.df_vid_points_imputed.loc[frame_counter])
					final_report_text += report_text

				elif frame_counter == self.jump_peak_frame:
					logging.debug(f"Analysing video - frame {frame_counter} of {self.total_frames} - JUMP PEAK posture frame.")
					skeleton_frame, peak_frame_text, report_text = self.jump_quality_at_peak_frame(frame, frame_counter, self.df_vid_points.loc[frame_counter], self.df_vid_points_imputed.loc[frame_counter])
					final_report_text += report_text

				elif frame_counter == self.jump_land_frame:
					logging.debug(f"Analysing video - frame {frame_counter} of {self.total_frames} - LANDING posture frame.")
					skeleton_frame, land_frame_text, report_text = self.jump_quality_at_land_frame(frame, frame_counter, self.df_vid_points.loc[frame_counter], self.df_vid_points_imputed.loc[frame_counter])
					final_report_text += report_text

				else:
					logging.debug(f"Analysing video - frame {frame_counter} of {self.total_frames}.")
					skeleton_frame = self.overlay_imputed_skeleton(frame, frame_counter, self.df_vid_points.loc[frame_counter], self.df_vid_points_imputed.loc[frame_counter])

				skeleton_frame = self.overlay_frame_text(skeleton_frame, squat_frame_text, peak_frame_text, land_frame_text)

				# Saving frame to output video using the VideoWriter object defined above.
				out.write(skeleton_frame)

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
		
		self.analysed_report_path = self.new_final_report_name(self.video_path)

		with open(self.analysed_report_path, "w") as text_file:
			text_file.write(final_report_text)
			
		print(f"Analysis report saved at {self.analysed_report_path}\n")
		logging.debug(f"Analysis report saved at '{self.analysed_report_path}'.")
