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

	my_video.create_frame_labels_df()
	logging.debug(f"Successfully created df_frame_labels.")

	my_video.analyse_vjump_video()


	print("Saving data...")
	my_video.save_data()
	logging.debug(f"Saved all data.")

	print("DONE!\nProgram will exit in 10 seconds.")
	time.sleep(10)

if __name__ == "__main__":
	
	logging.debug(f"----- MAIN PROGRAM STARTED -----\n")

	run_vjump_dataset_creation()

	logging.debug(f"----- MAIN PROGRAM FINISHED -----\n\n\n")