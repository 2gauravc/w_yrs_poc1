
###################################################################
#About this pipeline
#Input: Takes in a video from teh s3 bucket w-yrs-input-video
#
#Processing:
#1.Uses the fastai model to detect the critical pose in each frame
#2.Writes frame # and detected pose (e.g. Squat) on each frame
#3.Produces an output video in ./tmp folder (x_ANALYSIS_VIDEO)
#4 Compress the video using ffmpeg and saves it to ./tmp (x_ANALYSIS.mp4)
#5.Save the video onto S3 bucket
#
#Output:
#1. Places a processed mp4 video in bucket w-yrs-processed-video (x_ANALYSIS.mp4)
#2. Saves a list of frame labels (detected pose) in the VJump object (attribute labels)

###################################################################

from VJump import VJump
import cv2
import time
import logging
import sys
import argparse
import os
from scipy import ndimage
import boto3

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
    
def upload_file_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3')

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except ClientError as e:
        print("Error")
        return False

def download_file_from_s3(bucket,s3_file, local_file_path):
    print('Downloading file from S3...')
    s3 = boto3.resource('s3')

    try:
        s3.Bucket(bucket).download_file(s3_file, local_file_path)
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except ClientError as e:
        print("Error")
        return False


def vjump_create_video_with_critical_frame_identified(argv):

        #  Parse argv to get the video file
        ap = argparse.ArgumentParser()
        ap.add_argument("--file", required=True, help="video file to download and process")
        ap.add_argument("--orient", required=True, help="athlete orientation (left or right)")
        args = vars(ap.parse_args())
        video_file = args["file"]
        orientation = args["orient"]
        
        #Download video file from S3 to ./tmp/
    
        temp_dir = './tmp/'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        video_download_path = os.path.join(temp_dir, video_file)
        bucket = 'w-yrs-input-video'
        
        if not download_file_from_s3(bucket,video_file, video_download_path):
            print ("Could not download file from S3. Exiting..")
            sys.exit(1)
        
        # Check if this is a valid video file and cv2 can open it
        if not check_video_file_open(video_download_path):
            print ("CV2 cannot open this file. Not a valid video file. Exiting..")
            logging.debug(f"CV2 cannot open this file. Not a valid video file. Exiting..")
            sys.exit(1)
        
	    
        print("\nStarting Processing.. PLEASE WAIT!\n")
        
        
        my_video = VJump.VJump(video_download_path, orientation)
        logging.debug(f"VJump_data object successfully created.")
        
    
        
        ## Creating a VideoCapture object.
        cap = cv2.VideoCapture(my_video.video_path)
		
		# Capturing the fps
        fps = cap.get(cv2.CAP_PROP_FPS)
		
		## Creating an object to write the new video
        my_video.analysed_video_path = my_video.new_final_vid_name(my_video.video_path)

		# Getting the video frame width and height.
        my_video.frame_width = int(cap.get(3))
        my_video.frame_height = int(cap.get(4))

		
        num_frame = 0
        lbls = []
        out = None
		
		
		# Load the fastai model
        my_video.fastai('../efficientnet_lite0__v4.2.pkl')
 		
        # Start the frame by frame loop
        
        while(cap.isOpened()):

            # Grabbing each individual frame, frame-by-frame.
            ret, frame = cap.read()

            if ret==True:
                num_frame += 1
                rotated_frame=ndimage.rotate(frame,270)
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    my_video.analysed_video_path = my_video.new_final_vid_name(my_video.video_path)
                    (h, w) = rotated_frame.shape[:2]
                    out = cv2.VideoWriter(my_video.analysed_video_path, fourcc, fps, (w, h))
                
                # Detect the pose using fastai model
                lbl = my_video.fastai.predict(rotated_frame)[0]
                lbls.append(lbl)
                
                # Put the Frame number and label on the frame
                cv2.putText(rotated_frame, "Fr#: {}, Lbl: {}".format(num_frame, lbl), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)
 
                 
                out.write(rotated_frame)
				
                print ("Processed frame {}. Label is {}".format(num_frame, lbl))

                
                # Exiting if "Q" key is pressed on the keyboard.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
        
        
        # Save the attributes to VJump object
        my_video.total_frames = num_frame
        my_video.labels = lbls


		# Releasing the VideoCapture object.
        cap.release()
        out.release()
        
        # Shorten the video using ffmpeg
        vid_path, vid_name_ext = os.path.split(my_video.video_path)
        vid_name, vid_ext = os.path.splitext(vid_name_ext)
		
        new_vid_name_ext = vid_name + "_ANALYSIS"+".mp4"
        new_vid_path = os.path.join(vid_path, new_vid_name_ext)
        
        my_video.analysed_compressed_video_path = new_vid_path

        os.system("ffmpeg -i {} -vcodec libx264 -crf 20 {}".format(my_video.analysed_video_path,my_video.analysed_compressed_video_path))
        
        #Upload analyzed video to S3
        if upload_file_to_s3(my_video.analysed_compressed_video_path, 'w-yrs-processed-video', new_vid_name_ext):
            print ("All done.")
        else:
            print ("Could not upload analyzed video to S3. Saved analyzed video to: {}".format(my_video.analysed_video_path))

            
if __name__ == "__main__":
	
	logging.debug(f"----- MAIN PROGRAM STARTED -----\n")

	vjump_create_video_with_critical_frame_identified(sys.argv[1:])

	logging.debug(f"----- MAIN PROGRAM FINISHED -----\n\n\n")
