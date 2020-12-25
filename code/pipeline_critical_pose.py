
###################################################################
#About this pipeline
#Input: Takes in a video from the s3 bucket w-yrs-input-video
#
#Processing: in the follwoing steps
#1. Download the video from S3 bucket (input videos)
#2. Validate and open the file frame by frame (using CV2))
#3. Detect the critical pose using fastAI model and post-process (Writes frame # and detected pose)
#4. Compress the video using ffmpeg and saves it to ./tmp (x_ANALYSIS.mp4)
#5. Upload the analyzed video onto S3 bucket
#6. Download the actual tagging file from S3 bucket
#7. Parse the taggings (and note errors if any)
#8. Create a dataframe (row = frame) with actual and detected pose tagging
#9. Save the new dataframe as a csv on disk
#10.Upload the csv file to ESQL DB

#
#Output:
#1. Places a processed mp4 video in bucket w-yrs-processed-video (x_ANALYSIS.mp4)
#2. Saves a list of frame labels (detected pose) in the VJump object (attribute labels)

###################################################################

from VJump import VJump
from functions import general_functions as gf
from functions import db_functions as db

import cv2
import time
import logging
import sys
import argparse
import os
from scipy import ndimage
import boto3
import shutil
import pandas as pd


timestr = time.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f"../inputs_outputs/logs/vjump_dataset_creation_log_{timestr}.txt", filemode='a', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# logging.disable(logging.CRITICAL)


def vjump_create_video_with_critical_frame_identified(argv):

        #  Parse argv to get the video file
        ap = argparse.ArgumentParser()
        ap.add_argument("--file", required=True, help="video file to download and process")
        ap.add_argument("--rotate", required=True, help="athlete orientation (left or right)")
        args = vars(ap.parse_args())
        video_file = args["file"]
        rotate = int(args["rotate"])
        
        files_written=[]
        
        
        #Step1. Download video file from S3
        #-------------------------------------------------------------------------------------------
        print("Starting Step 1. Download video file from S3\n")
        temp_dir = './tmp/'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        video_download_path = os.path.join(temp_dir, video_file)
        bucket = 'w-yrs-input-video'
        
        if not gf.download_file_from_s3(bucket,video_file, video_download_path):
            print ("Could not download file from S3. Exiting..")
            sys.exit(1)
        
        files_written.append(video_download_path)
        
        # Check if this is a valid video file and cv2 can open it
        if not gf.check_video_file_open(video_download_path):
            print ("CV2 cannot open this file. Not a valid video file. Exiting..")
            logging.debug(f"CV2 cannot open this file. Not a valid video file. Exiting..")
            sys.exit(1)
        
	    
	    # Step 2. Validate and open the file frame by frame
	    #----------------------------------------------------------------------
	    
        print("Starting Step 2. Validate and open the file frame by frame\n")
        
        orientation = 'left' ## This is redundant
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
        confs =[]
        out = None
		
		
		# Load the fastai model
        my_video.fastai('../efficientnet_lite0__v4.2.pkl')
 		
        # Start the frame by frame loop
        
        print("Starting Step 3. Detect the critical pose \n")
        
        while(cap.isOpened()):

            # Grabbing each individual frame, frame-by-frame.
            ret, frame = cap.read()

            if ret==True:
                num_frame += 1
                if rotate == 0:
                    rotated_frame = frame
                else:
                    rotated_frame=ndimage.rotate(frame,rotate)
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    my_video.analysed_video_path = my_video.new_final_vid_name(my_video.video_path)
                    (h, w) = rotated_frame.shape[:2]
                    out = cv2.VideoWriter(my_video.analysed_video_path, fourcc, fps, (w, h))
                
                # Step3. Detect the critical pose using fastai model and postprocess
                #----------------------------------------------------------------------
                pred = my_video.fastai.predict(rotated_frame)
                label_idx = int(pred[1].numpy())
                conf = round(pred[2].numpy()[label_idx].item(),2)
                lbl = pred[0]
                
                lbls.append(lbl)
                confs.append(conf)
                
                # Put the Frame number and label on the frame
                cv2.putText(rotated_frame, "Fr#: {}, Lbl: {}".format(num_frame, lbl), (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)
 
                 
                out.write(rotated_frame)
				
                print ("Processed frame {}. Label is {}. Conf is {}".format(num_frame, lbl, conf))

                
                # Exiting if "Q" key is pressed on the keyboard.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break
        
        
        # Save the attributes to VJump object
        my_video.total_frames = num_frame
        my_video.labels = lbls
        my_video.confs = confs


		# Releasing the VideoCapture object.
        cap.release()
        out.release()
        
        #Step 4. Compress the video using ffmpeg
        #------------------------------------------------------------------------
        print("Starting Step 4. Compress the video using ffmpeg\n")
        vid_path, vid_name_ext = os.path.split(my_video.video_path)
        vid_name, vid_ext = os.path.splitext(vid_name_ext)
		
        new_vid_name_ext = vid_name + "_ANALYSIS"+".mp4"
        new_vid_path = os.path.join(vid_path, new_vid_name_ext)

        # compress_video
        ret_val = gf.compress_video_ffmpeg(my_video.analysed_video_path, new_vid_path)
        
        if ret_val != 0:
            print ("Video Compression failed. Continuing with original video")
            shutil.copyfile(my_video.analysed_video_path, new_vid_path)
        
        files_written.append(new_vid_path)
        
        my_video.analysed_compressed_video_path = new_vid_path
                
        
        #Step 5. Upload analyzed video to S3
        #-----------------------------------------------------------------------------
        print("Starting Step 5. Upload analyzed video to S3\n")
        if gf.upload_file_to_s3(my_video.analysed_compressed_video_path, 'w-yrs-processed-video', new_vid_name_ext):
            print ("All done.")
        else:
            print ("Could not upload analyzed video to S3. Saved analyzed video to: {}".format(my_video.analysed_video_path))
            
        # Step 6. Download the actual pose (ground truth) tagging from the S3 bucket
        #--------------------------------------------------------------------------
        print("Starting Step 6. Download the actual pose (ground truth) tagging from the S3 bucket\n")
        ## Construct the name of the csv file with the tagging
        
        actual_pose_filename = vid_name + "_FRAME_LABELS.csv"
        bucket = 'w-yrs-pose-detect-labels'
        
        file_download_path = os.path.join(temp_dir, actual_pose_filename)
        
        if not gf.download_file_from_s3(bucket,actual_pose_filename, file_download_path):
            print ("Could not download tag file from S3. Exiting..")
            sys.exit(1)
        
        files_written.append(file_download_path)
        
        # Step 7. Parse the taggings (and note errors if any)
        #------------------------------------------------------------------------
        print("Starting Step 7. Parse the taggings (and note errors if any)\n")
        frames, frames_tag_list = gf.parse_tags (file_download_path) #tagging is changed to 'error' if error suspected in frame tagging
        
        
        # Generate the actual_pose vector by frame number
        num_recs = len(frames)
        
        if (num_recs != my_video.total_frames): #something is wrong
            print ("Number of video frames detected by CV2 do not match the pose tag file. Exiting...")
            sys.exit(1)
        
        
        # Step 8. Create a dataframe (row = frame) with actual and detected pose tagging
        #---------------------------------------------------------------------------------
        print("Starting Step 8. Parse the taggings (and note errors if any)\n")
        videofileName = [vid_name_ext for i in range(num_recs)]
        frame_no = frames
        model_name = ["fastaipose" for i in range(num_recs)]
        model_version = ["v1" for i in range(num_recs)]
        detected_pose = my_video.labels
        detected_pose_conf = my_video.confs
        actual_pose = frames_tag_list
        
        
        df = pd.DataFrame ({'videofilename':videofileName,'frame_no':frame_no, 'model_name':model_name, 'model_version':model_version, 'detected_pose':detected_pose,'detected_pose_conf':detected_pose_conf, 'actual_pose':actual_pose})
        
        
        # Step 9. Save the new dataframe as a csv on disk
        #------------------------------------------------------------------------------------
        print("Starting Step 9. Save the new dataframe as a csv on disk\n")
        file_path, file_name_ext = os.path.split(file_download_path)
        file_name, file_ext = os.path.splitext(file_name_ext)
        new_file_name_ext = file_name + "_T"+ file_ext
        new_file_path = os.path.join(file_path, new_file_name_ext)
        
        df.to_csv(new_file_path, index=False, header=False)
        
        files_written.append(new_file_path)
        
        #Step 10. Upload the csv file to ESQL DB
        #------------------------------------------------------------------------------------
        print("Starting Step 10. Upload the csv file to ESQL DB\n")
        
        db_table = 'video_frame_vjump_pose'
        #Check the number of records in the table
        db.report_table_recs([])
        
        db.upload_csv_to_db(new_file_path, db_table)
        
        
        #Step 11. Clean-up (remove temp files etc.)
        print("Starting Step 11. Clean-up (remove temp files etc.)\n")
        num_removed = gf.remove_temp_files(files_written)
        print("{} files removed".format(num_removed))
            
if __name__ == "__main__":
	
	logging.debug(f"----- MAIN PROGRAM STARTED -----\n")

	vjump_create_video_with_critical_frame_identified(sys.argv[1:])

	logging.debug(f"----- MAIN PROGRAM FINISHED -----\n\n\n")
