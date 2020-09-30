from VJump import VJump
import cv2
import time
import logging

logging.basicConfig(filename="../inputs_outputs/logs/vjump_log.txt", filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# logging.disable(logging.CRITICAL)

def run_vjump_analysis():

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

	correct_orientation_input = False

	while correct_orientation_input == False:
		orientation = input("Is the person in the video facing towards left or right of the video frame?\n\nINPUT-> ")
		logging.debug(f"Orientation given by user: '{orientation}'.")
		orientation = orientation.lower()

		if orientation not in ["right", "left"]:
			print("Enter only 'left' or 'right'.\n")
			logging.debug(f"Orientation '{orientation}' not accepted as input.")
		else:
			correct_orientation_input = True
			logging.debug(f"Orientation '{orientation}' accepted as input.")

	print("\nPROCESSING... PLEASE WAIT!\n")

	my_video = VJump.VJump(video_path, orientation)
	logging.debug(f"VJump object successfully created.")

	logging.debug(f"Starting to count total frames.")
	my_video.count_frames()
	print(f"Total frames in video = {my_video.total_frames}")
	logging.debug(f"Total frames in video = {my_video.total_frames}")

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
		print("Identifying critical frames in video...\n")
		logging.debug(f"Starting to identify critical frames in video.")
		my_video.identify_critical_frames()

		print("CRITICAL FRAMES IDENTIFIED:")
		print(f"Squat posture: frame {my_video.jump_squat_frame}")
		print(f"Jump Peak posture: frame {my_video.jump_peak_frame}")
		print(f"Landing posture: frame {my_video.jump_land_frame}\n")

		logging.debug(f"Successfully identified critical frames in video.")
		logging.debug(f"Squat posture frame = {my_video.jump_squat_frame}")
		logging.debug(f"Jump Peak posture frame = {my_video.jump_peak_frame}")
		logging.debug(f"Landing posture frame = {my_video.jump_land_frame}")

	except:
		print("Error while identifying critical frames in video. Please provide a new video.\n")
		print("Program will exit in 10 seconds.")
		logging.exception(f"EXCEPTION OCCURED - Unable to identify critical frames in video. Program will be TERMINATED.")
		time.sleep(10)
		return

	logging.debug(f"Starting to access vertical jump success criteria file.")
	my_video.import_vjump_criteria()
	logging.debug(f"Successfully accessed vertical jump success criteria file.")

	logging.debug(f"Starting to assess need for data imputation in critical frames.")
	if my_video.need_to_impute_critical_frames() == False:
		logging.debug(f"Successfully assessed need for data imputation. Imputation NOT NEEDED.")
		print("CRITICAL FRAMES ADJUSTED:")
		print(f"Squat posture: frame {my_video.jump_squat_frame}")
		print(f"Jump Peak posture: frame {my_video.jump_peak_frame}")
		print(f"Landing posture: frame {my_video.jump_land_frame}\n")
	else:
		logging.debug(f"Successfully assessed need for data imputation. Imputation NEEDED.")
		print("Critical frames identified above have missing body points. Frames around these critical frames also have missing body points. So, these missing body points will now be imputed.\n")
		logging.debug(f"Starting to perform data imputation for all frames in the video.")
		my_video.impute_missing_body_points()
		logging.debug(f"Successfully performed data imputation.")

	try:
		print("Analysing video according to vertical jump criteria...")
		logging.debug(f"Starting to analyse video according to vjump criteria.")
		my_video.analyse_vjump_video()
		logging.debug(f"Successfully analysed video according to vjump criteria.")
	except:
		print("Error while analysing video.\n")
		print("Program will exit in 10 seconds.")
		logging.exception(f"EXCEPTION OCCURED - Unable to analyse video. Program will be TERMINATED.")
		time.sleep(10)
		return

if __name__ == "__main__":
	
	logging.debug(f"----- MAIN PROGRAM STARTED -----\n")

	run_vjump_analysis()

	logging.debug(f"----- MAIN PROGRAM FINISHED -----\n\n\n")