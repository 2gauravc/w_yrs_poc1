from VJump import VJump_dataset_creation
import cv2
import time
import logging

timestr = time.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f"../inputs_outputs/logs/vjump_dataset_creation_log_{timestr}.txt", filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# logging.disable(logging.CRITICAL)

def run_vjump_dataset_creation():

	# logging.debug(f"----- MAIN PROGRAM STARTED -----\n")

	video_read_successfully = False
	attempt = 1

	while video_read_successfully == False and attempt <= 5:
		video_path = input("Provide the video filepath.\n\nINPUT-> ")
		logging.debug(f"Video path given by user: '{video_path}'.")
		print("\n")
		
		try:
			cap = cv2.VideoCapture(video_path)

			# Checking if video can be accessed successfully.
			if (cap.isOpened() == False):
				print("Unable to read video. Re-enter video path.")
				print(f"Attempts remaining: {5-attempt}")
				logging.debug(f"Attempt {attempt} - unable to access video.")
				attempt += 1
			else:
				video_read_successfully = True
				logging.debug(f"Attempt {attempt} - video accessed successfully.")

			# Releasing the VideoCapture object.
			cap.release()

		except:
			print("Unable to read video. Re-enter video path.")
			print(f"Attempts remaining: {5-attempt}")
			logging.exception(f"EXCEPTION OCCURED - Attempt {attempt} - unable to access video.")
			attempt += 1

	if video_read_successfully == False:
		print("Unable to read video.")
		print("Program will exit in 10 seconds.")
		logging.debug(f"Unable to access video after 5 attempts. Program will be TERMINATED.")
		time.sleep(10)
		return
	
	print("\nPROCESSING... PLEASE WAIT!\n")

	my_video = VJump_dataset_creation.VJump_data(video_path)
	logging.debug(f"VJump_data object successfully created.")

	logging.debug(f"Starting to count total frames and save them.")
	my_video.count_and_save_frames()
	print(f"Total frames in video = {my_video.total_frames}")
	logging.debug(f"Total frames in video = {my_video.total_frames}")

	"""
	try:
		logging.debug(f"Starting to import body net model files.")
		my_video.import_body_net()
		logging.debug(f"Successfully imported body net model files.")
	except:
		print("Could not find body net model files to find body points in video. Ensure that files pose_deploy.prototxt & pose_iter_584000.caffemodel are stored in ../inputs_outputs/models folder.\n")
		print("Program will exit in 10 seconds.")
		logging.exception(f"EXCEPTION OCCURED - Unable to access body net model files. Program will be TERMINATED.")
		time.sleep(10)
		return

	try:
		logging.debug(f"Starting to locate body points in video.")
		my_video.locate_body_points()
		logging.debug(f"Successfully located body points in video.")
	except:
		print("Error while locating body points on video. Please provide a new video.\n")
		print("Program will exit in 10 seconds.")
		logging.exception(f"EXCEPTION OCCURED - Unable to locate body points in video. Program will be TERMINATED.")
		time.sleep(10)
		return
	
	try:
		print("Normalizing body points...")
		logging.debug(f"Starting to normalize body points.")
		my_video.normalize_body_points()
		print("Successfully normalized body points.\n")
		logging.debug(f"Successfully normalized body points.")

	except:
		print("Error while normalizing body points. Please provide a new video.\n")
		print("Program will exit in 10 seconds.")
		logging.exception(f"EXCEPTION OCCURED - Unable to normalize body points. Program will be TERMINATED.")
		time.sleep(10)
		return

	my_video.df_vid_points = my_video.separate_x_y_points(my_video.df_vid_points)
	logging.debug(f"Successfully separated x & y coordinates in df_vid_points.")

	my_video.df_vid_points_norm = my_video.separate_x_y_points(my_video.df_vid_points_norm)
	logging.debug(f"Successfully separated x & y coordinates in df_vid_points_norm.")

	"""

	my_video.create_frame_labels_df()
	logging.debug(f"Successfully created df_frame_labels.")

	confirm_squat_frames_input = False

	while confirm_squat_frames_input == False:
		squat_frames_input = input("Enter frame numbers where person is in SQUAT POSTURE. (Enter numbers separated by commas, 0 if no frames show squat posture.)\n\nINPUT-> ")
		logging.debug(f"SQUAT POSTURE frames given by user: {squat_frames_input}.")
		squat_frames_input = my_video.convert_frame_labels_input(squat_frames_input)

		if my_video.correct_frame_labels_input(squat_frames_input):

			confirm_input = ""
			
			while confirm_input not in ["y", "n"]:
				confirm_input = input(f"Confirm SQUAT POSTURE frames : {squat_frames_input}\n\ny/n? -> ")
				confirm_input = confirm_input.lower()

				if confirm_input == "y":
					confirm_squat_frames_input = True
					logging.debug(f"User confirmed SQUAT POSTURE frames {squat_frames_input}.")
					my_video.confirm_frames_input("squat", squat_frames_input)

				elif confirm_input == "n":
					logging.debug(f"User denied confirmation of SQUAT POSTURE frames {squat_frames_input}.")
				
				else:
					print("Enter only 'y' or 'n'.\n")
					logging.debug(f"User entered incorrect confirmation : '{confirm_input}'.")

		else:
			print(f"Frame numbers should only be between [1 - {my_video.total_frames}].")
			logging.debug(f"One or more frames entered are outside range [1 - {my_video.total_frames}].")

	confirm_jump_peak_frames_input = False

	while confirm_jump_peak_frames_input == False:
		jump_peak_frames_input = input("Enter frame numbers where person is in JUMP PEAK POSTURE. (Enter numbers separated by commas, 0 if no frames show jump peak posture.)\n\nINPUT-> ")
		logging.debug(f"JUMP PEAK POSTURE frames given by user: {jump_peak_frames_input}.")
		jump_peak_frames_input = my_video.convert_frame_labels_input(jump_peak_frames_input)

		if my_video.correct_frame_labels_input(jump_peak_frames_input):

			confirm_input = ""
			
			while confirm_input not in ["y", "n"]:
				confirm_input = input(f"Confirm JUMP PEAK POSTURE frames : {jump_peak_frames_input}\n\ny/n? -> ")
				confirm_input = confirm_input.lower()

				if confirm_input == "y":
					confirm_jump_peak_frames_input = True
					my_video.confirm_frames_input("jump_peak", jump_peak_frames_input)
					logging.debug(f"User confirmed JUMP PEAK POSTURE frames {jump_peak_frames_input}.")
				elif confirm_input == "n":
					logging.debug(f"User denied confirmation of JUMP PEAK POSTURE frames {jump_peak_frames_input}.")
				else:
					print("Enter only 'y' or 'n'.\n")
					logging.debug(f"User entered incorrect confirmation : '{confirm_input}'.")

		else:
			print(f"Frame numbers should only be between [1 - {my_video.total_frames}].")
			logging.debug(f"One or more frames entered are outside range [1 - {my_video.total_frames}].")

	confirm_landing_frames_input = False

	while confirm_landing_frames_input == False:
		landing_frames_input = input("Enter frame numbers where person is in LANDING POSTURE. (Enter numbers separated by commas, 0 if no frames show landing posture.)\n\nINPUT-> ")
		logging.debug(f"LANDING POSTURE frames given by user: {landing_frames_input}.")
		landing_frames_input = my_video.convert_frame_labels_input(landing_frames_input)

		if my_video.correct_frame_labels_input(landing_frames_input):

			confirm_input = ""
			
			while confirm_input not in ["y", "n"]:
				confirm_input = input(f"Confirm LANDING POSTURE frames : {landing_frames_input}\n\ny/n? -> ")
				confirm_input = confirm_input.lower()

				if confirm_input == "y":
					confirm_landing_frames_input = True
					my_video.confirm_frames_input("landing", landing_frames_input)
					logging.debug(f"User confirmed LANDING POSTURE frames {landing_frames_input}.")
				elif confirm_input == "n":
					logging.debug(f"User denied confirmation of LANDING POSTURE frames {landing_frames_input}.")
				else:
					print("Enter only 'y' or 'n'.\n")
					logging.debug(f"User entered incorrect confirmation : '{confirm_input}'.")

		else:
			print(f"Frame numbers should only be between [1 - {my_video.total_frames}].")
			logging.debug(f"One or more frames entered are outside range [1 - {my_video.total_frames}].")

	print("Saving data...")
	my_video.save_data()
	logging.debug(f"Saved all data.")

	try:
		print("Creating zip file...")
		my_video.zip_data()
		logging.debug(f"Successfully created zip file at '{my_video.zip_path}'.")
	except:
		print("Error while creating zip file. Please create a zip file manually. All data is saved in the same folder as the video.\n")
		print("Program will exit in 10 seconds.")
		logging.exception(f"EXCEPTION OCCURED - Unable to create zip file. Program will be TERMINATED.")
		time.sleep(10)
		return

	print("DONE!\nProgram will exit in 10 seconds.")
	time.sleep(10)

if __name__ == "__main__":
	
	logging.debug(f"----- MAIN PROGRAM STARTED -----\n")

	run_vjump_dataset_creation()

	logging.debug(f"----- MAIN PROGRAM FINISHED -----\n\n\n")