from VJump import VJump
import cv2
import time
import logging
import sys
import argparse
import os

timestr = time.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f"../inputs_outputs/logs/vjump_dataset_creation_log_{timestr}.txt", filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# logging.disable(logging.CRITICAL)
def check_video_file_open(video_path):
    video_read_successfully = False

    try:
        cap = cv2.VideoCapture(video_path)
        if (cap.isOpened() == True):
            video_read_successfully = True
        else:
            print ("Unable to read video")
    except:
        print ("Exception in trying to read video file")

    finally:
        cap.release()
        return (video_read_successfully)


def vjump_create_video_with_critical_frame_identified(argv):

        #  Parse argv to get the video file 
        ap = argparse.ArgumentParser()
        ap.add_argument("--videofile", required=True, help="vjump video file path")
        ap.add_argument("--orient", required=True, help="athlete orientation (left or right)")
        args = vars(ap.parse_args())
        video_path = args["videofile"]
        orientation = args["orient"]
        
        print ("Picking up video file {}".format(video_path))
        if os.path.exists(video_path) == False:
            print("Video file not found. Exiting...")
            logging.debug(f"Video file not found.Exiting..")
            sys.exit(1)
        
        # Check if this is a valid video file and cv2 can open it
        if not check_video_file_open(video_path):
            print ("CV2 cannot open this file. Not a valid video file. Exiting..")
            logging.debug(f"CV2 cannot open this file. Not a valid video file. Exiting..")
            sys.exit(1)
        
	
        print("\nStarting Processing.. PLEASE WAIT!\n")

        my_video = VJump.VJump(video_path, orientation)
        logging.debug(f"VJump_data object successfully created.")

        logging.debug(f"Starting to count total frames.")
        my_video.count_and_save_frames()
        print(f"Total frames in video = {my_video.total_frames}")
        logging.debug(f"Total frames in video = {my_video.total_frames}")

	# Load the fastai model to detect critical poses 
        my_video.fastai('../efficientnet_lite0__v4.2.pkl')

        my_video.estimate_key_pose() 

        #my_video.create_frame_labels_df()
	#logging.debug(f"Successfully created df_frame_labels.")

	#my_video.analyse_vjump_video()


	#print("Saving data...")
	#my_video.save_data()
	#logging.debug(f"Saved all data.")

        print("DONE!\nProgram will exit in 10 seconds.")
        time.sleep(10)

if __name__ == "__main__":
	
	logging.debug(f"----- MAIN PROGRAM STARTED -----\n")

	vjump_create_video_with_critical_frame_identified(sys.argv[1:])

	logging.debug(f"----- MAIN PROGRAM FINISHED -----\n\n\n")
