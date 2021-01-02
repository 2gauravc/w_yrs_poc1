# This is the pipeline to find pose points for critical frames and evaluate each pose
# Input:
#  1. Video File
# Output:
#  1. Pose points (for all frames) uploaded to database on ESQL
# Processing:
#  1. Download the video file from S3
#  2. Download the pose point detection model from S3
#  3. Perform pose point detection on all the frames (one by one)
#  4. Upload the detected pose points to ESQL DB (video_pose_proc_summary, video_pose_proc_details) (option -d: this action is done only when the '-d' option is used)
#  5. Upload the pose point tagged video onto S3 (option -o: this action is done only when the '-o' option is used)

from functions import general_functions as gf
from functions import db_functions as db
from functions import tf_pose_functions as tf
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import cv2
import time
from datetime import datetime
from scipy import ndimage

import sys, os, argparse


def main_human_pose_pipeline(argv):
    #  Parse argv to get the video file and the rotate angle (if needed)
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="video file to download and process")
    ap.add_argument("--rotate", required=True, help="degrees to rotate the image extracted from video")
    ap.add_argument('-o', '--out', action='store_true', help="write video output to S3")
    ap.add_argument('-d', '--dbwrite', action='store_true', help="write critical pose output to DB")
    args = vars(ap.parse_args())
    video_file = args["file"]
    rotate = int(args["rotate"])
    wr_s3 = args["out"]
    wr_db = args["dbwrite"]

    vid_name, vid_ext = os.path.splitext(video_file)
    
    #Step 1. Download the video file from S3
    print("Starting Step 1. Download video file from S3\n")
    
    temp_dir = './tmp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    
    video_download_path = os.path.join(temp_dir, video_file)
    bucket = 'w-yrs-input-video'
    
    if not gf.download_file_from_s3(bucket,video_file, video_download_path):
        print ("Could not download video file from S3. Exiting..")
        sys.exit(1)
    

    #Step 2. Download the human pose model from S3
    print("Starting Step 2. Download pose model from S3\n")
    model_file = 'posenet_v1.tflite'
    model_download_path = os.path.join(temp_dir, model_file)
    bucket = 'w-yrs-models'
    
    if not gf.download_file_from_s3(bucket,model_file, model_download_path):
        print ("Could not download model file from S3. Exiting..")
        sys.exit(1)
    
    
    #Step 3. Perform pose point detection on all the frames (one by one)
    print("Starting Step 3. Pose point detection (frame by frame)\n")
    
    #orientation = 'left' ## This is redundant
    #my_video = VJump.VJump(video_download_path, orientation)
    
    cap = cv2.VideoCapture(video_download_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Getting the video frame width and height.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    num_frame=0
    frame_nos = []
    std_kps = []
    kps = []
    out = None
    
    start = time.time()
    
    #frames_to_print = [17,41,42]
    
    while(cap.isOpened()):
        # Grabbing each individual frame, frame-by-frame.
        ret, frame = cap.read()

        if ret==True:
            num_frame += 1
            if rotate == 0:
                rotated_frame = frame
            else:
                rotated_frame=ndimage.rotate(frame,rotate)
            #Save the image to temp_dir
            temp_image_name = vid_name + "_FRAME_" + str(num_frame) + ".png"
            temp_image_path = os.path.join(temp_dir, temp_image_name)
            cv2.imwrite(temp_image_path, rotated_frame)
        
            #Perform pose point estimation on the rotated frame
            kp, image_show = tf.get_pose_points_tf(model_download_path, temp_image_path)
        
            # Append the frame number and std KPs
            frame_nos.append(num_frame)
            kps.append(kp)
            
            #if num_frame in frames_to_print:
            #    print("KP for Frame no. {}".format(num_frame))
            #    print(kp)
            
            #write to S3 if needed
            if wr_s3:
                pose_img = tf.draw_kps(image_show.copy(),kp)
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    analysed_video_path = os.path.join(temp_dir, vid_name+"_POSE_POINTS.mp4")
                    (h, w) = pose_img.shape[:2]
                    out = cv2.VideoWriter(analysed_video_path, fourcc, fps, (w, h))
                
                
                out.write(pose_img)
                 
            
            # Delete the temp image file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        else:
            break
    
    cap.release()
    if out is not None:
        out.release()
    
    end = time.time()
    total_frames = num_frame
    print ("Pose point model processed {} frames".format(total_frames))
    proc_time = (end - start)
    # Prepare the video
    
    time_now =  datetime.now().strftime("%d/%m/%Y %H:%M")
        
    #Create the video_pose_proc_summary dataset
    video_file_name = video_file
    model_name = "posenet_tflite"
    model_version = "v1"
    total_frames = total_frames
    total_processing_time_secs = round(proc_time,0)
    FPS = total_frames / total_processing_time_secs
    processed_on = time_now
    
    
    df_pose = tf.create_pose_points_df(video_file_name, total_frames, frame_nos,model_name, model_version, kps)
    #print (df_pose)
    
    df_pose_path = os.path.join(temp_dir, vid_name+"_POSE_POINTS.csv")
    col_list = ['video_file_name', 'frame_num', 'model_name', 'model_version', 'confidence','pt0_forehead_x', 'pt0_forehead_y', 'pt1_upperchest_x', 'pt1_upperchest_y', 'pt2_rightshoulder_x',
    'pt2_rightshoulder_y', 'pt3_rightelbow_x', 'pt3_rightelbow_y', 'pt4_rightwrist_x', 'pt4_rightwrist_y', 'pt5_leftshoulder_x', 'pt5_leftshoulder_y', 'pt6_leftelbow_x',
    'pt6_leftelbow_y', 'pt7_leftwrist_x', 'pt7_leftwrist_y', 'pt8_navel_x', 'pt8_navel_y', 'pt9_righthip_x', 'pt9_righthip_y', 'pt10_rightknee_x', 'pt10_rightknee_y',
    'pt11_rightankle_x', 'pt11_rightankle_y', 'pt12_lefthip_x', 'pt12_lefthip_y', 'pt13_leftknee_x', 'pt13_leftknee_y', 'pt14_leftankle_x', 'pt14_leftankle_y']
    df_pose.reindex(columns = col_list).to_csv(df_pose_path, index=False, header = False)
    #print(df_pose.columns)
    
    #Step: 4. Upload the detected pose points to ESQL DB (video_pose_proc_details)
    if wr_db:
        print("Starting Step 4. Upload the detected pose points to ESQL DB\n")
        
        # Write the video_pose_proc_summary dataset to ESQL DB
        con = db.connect_db()
        cur = con.cursor()

        postgres_insert_query = """ INSERT INTO video_pose_proc_summary (video_file_name, model_name, model_version, total_frames, total_processing_time_secs, FPS) VALUES (%s,%s,%s,%s, %s, %s)"""
        record_to_insert = (video_file_name, model_name,model_version, total_frames,total_processing_time_secs, FPS)
        cur.execute(postgres_insert_query, record_to_insert)
        con.commit()

        count = cur.rowcount
        print (count, "Record inserted successfully into video_pose_proc_summary table")
        con.close()

        db_table = 'video_pose_proc_details'
        #Check the number of records in the table
        db.report_table_recs([db_table])
        db.upload_csv_to_db(df_pose_path, db_table)
        db.report_table_recs([db_table])

    else:
        print("Skipping Step 4. Upload the detected pose points to ESQL DB\n")

    
    
    
    # Step 5: Upload the pose point tagged video onto S3
    
    if wr_s3:
        print("Starting Step 5. Upload analyzed video to S3\n")
        if gf.upload_file_to_s3(analysed_video_path, 'w-yrs-processed-video', 'human_pose/'+vid_name+"_ANALYSIS.mp4"):
            print ("All done.")
        else:
            print ("Could not upload analyzed video to S3. Saved analyzed video to: {}".format(analysed_video_path))
    else:
        print("Skipping Step 5. Upload analyzed video to S3\n")
    

if __name__ == "__main__":
    main_human_pose_pipeline(sys.argv[1:])

