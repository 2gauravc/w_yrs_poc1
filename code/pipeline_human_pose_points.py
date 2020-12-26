# This is the pipeline to find pose points for critical frames and evaluate each pose
# Input:
#  1. Video File
# Output:
#  1. Pose points (for critical frames) uploaded to database on ESQL
#  2. VJUMP Report card generated and uploaded to S3
# Processing:
#  1. Download the file from S3
#  2. Query the critical frames from ESQL DB (table: video_frame_vjump_pose)
#  3. Perform pose point detection on critical frames
#  4. Identify frames with available pose points for evaluation
#  5. Evaluate frames and find the top ranked frame (per criticla pose)
#  6. Create the VJUMP report card
#  7. Upload the report card to S3

from functions import general_functions as gf
from functions import db_functions as db

import sys, os, argparse

def main_human_pose_pipeline(argv):
    #  Parse argv to get the video file and the rotate angle (if needed)
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="video file to download and process")
    ap.add_argument("--rotate", required=True, help="degrees to rotate the image extracted from video")
    args = vars(ap.parse_args())
    video_file = args["file"]
    rotate = int(args["rotate"])

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
    
    #Step 2. Query the critical frames from ESQL DB
    print("Starting Step 2. Query the critical frames from ESQL DB\n")
    
    model_name = 'fastaipose'
    model_version = 'v1'
    
    critical_frames_df = db.get_critical_frames(video_file, model_name, model_version, 0.70)
    
    
    
    




if __name__ == "__main__":
    main_human_pose_pipeline(sys.argv[1:])

