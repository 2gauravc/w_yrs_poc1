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
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np


import sys, os, argparse

def parse_pose_output(heatmap_data,offset_data, threshold):

  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''

  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):

      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1

  return pose_kps

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
    
    if (critical_frames_df.shape[0] == 0):
        print ("Cannot find critical frames for {} in db. Exiting.")
        sys.exit(1)
    
    
    #Step 3.Perform pose point detection on critical frames
    
    # Download the model from S3
    model_file = 'posenet_v1.tflite'
    model_download_path = os.path.join(temp_dir, model_file)
    bucket = 'w-yrs-models'
    
    if not gf.download_file_from_s3(bucket,model_file, model_download_path):
        print ("Could not download model file from S3. Exiting..")
        sys.exit(1)

    tflite_interpreter = tflite.Interpreter(model_path=model_download_path)
    tflite_interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    
    #print("Type of input:", input_details[0]['dtype'])
    
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    print("Height {}, Width {}".format(height, width))
    img = Image.open('./tmp/IMG_3770_FRAME_65.png').resize((width, height))
    
    # add N dim
    input_data = np.expand_dims(img, axis=0)
    
    input_mean = 127.5
    input_std = 127.5
    
    input_data = (np.float32(input_data) - input_mean) / input_std
    tflite_interpreter.set_tensor(input_details[0]['index'], input_data)

    #start_time = time.time()
    tflite_interpreter.invoke()
    #stop_time = time.time()

    output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
    offset_data = tflite_interpreter.get_tensor(output_details[1]['index'])
    
    # Getting rid of the extra dimension
    template_heatmaps = np.squeeze(output_data)
    template_offsets = np.squeeze(offset_data)
    
    print("template_heatmaps' shape:", template_heatmaps.shape)
    print("template_offsets' shape:", template_offsets.shape)
    
    template_kps = parse_pose_output(template_heatmaps,template_offsets,0.3)
    
    print(template_kps)


if __name__ == "__main__":
    main_human_pose_pipeline(sys.argv[1:])

