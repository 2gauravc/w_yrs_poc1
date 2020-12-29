from functions import general_functions as gf
from functions import db_functions as db
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from collections import OrderedDict


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

def draw_kps(show_img,kps, ratio=None):
    for i in range(5,kps.shape[0]):
      if kps[i,2]:
        if isinstance(ratio, tuple):
          cv2.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
          continue
        cv2.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img

def prep_kps_df(kps, ratio=None):
    x_coords =[]
    y_coords =[]
    
    for i in range(kps.shape[0]):
      if kps[i,2]:
        if isinstance(ratio, tuple):
            x = int(round(kps[i,1]*ratio[1]))
            y = int(round(kps[i,0]*ratio[0]))
            x_coords.append(x)
            y_coords.append(y)
        else:
            x = kps[i,1]
            y = kps[i,0]
            x_coords.append(x)
            y_coords.append(y)
            
    pose_body_points = ['nose','leftEye','rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
    'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']
    
    pose_points_df = pd.DataFrame({'body_points': pose_body_points, 'x':x_coords, 'y':y_coords})

    return (pose_points_df)

    
def get_pose_points_tf(model_path, image_path):

    tflite_interpreter = tflite.Interpreter(model_path=model_path)
    tflite_interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    
    #print("Type of input:", input_details[0]['dtype'])
    
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    #print("Height {}, Width {}".format(height, width))
    img = Image.open(image_path).resize((width, height))
    
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
    
    template_kps = parse_pose_output(template_heatmaps,template_offsets,0.3)
    
    template_show = np.squeeze((input_data.copy()*127.5+127.5)/255.0)
    template_show = np.array(template_show*255,np.uint8)

    return(template_kps, template_show)

def calc_conf_tf_pose(kps, total_frames):
    confidence = []
    for i in range(total_frames):
        l = len(kps[i])
        conf_frame = np.mean([kps[i][row,2] for row in range(l)])
        confidence.append(conf_frame)
    confidence = [round(num,2) for num in confidence]
    return (confidence)

def create_pose_points_df(video_file, total_frames, frame_nos,model_name, model_version, kps):
    #Create the video_pose_proc_details dataset
    video_file_name = [video_file for i in range(total_frames)]
    frame_num = frame_nos
    model_name_s = [model_name for i in range(total_frames)]
    model_version_s = [model_version for i in range(total_frames)]
    confidence = calc_conf_tf_pose(kps, total_frames)
    pt0_forehead_x = [-1 for i in range(total_frames)]
    pt0_forehead_y = [-1 for i in range(total_frames)]
    pt1_upperchest_x = [-1 for i in range(total_frames)]
    pt1_upperchest_y = [-1 for i in range(total_frames)]
    pt2_rightshoulder_x = [kps[i][6,0] if kps[i][6,2] == 1 else -1 for i in range(total_frames)]
    pt2_rightshoulder_y = [kps[i][6,1] if kps[i][6,2] == 1 else -1 for i in range(total_frames)]
    pt3_rightelbow_x = [kps[i][8,0] if kps[i][8,2] == 1 else -1 for i in range(total_frames)]
    pt3_rightelbow_y = [kps[i][8,1] if kps[i][8,2] == 1 else -1 for i in range(total_frames)]
    pt4_rightwrist_x = [kps[i][10,0] if kps[i][10,2] == 1 else -1 for i in range(total_frames)]
    pt4_rightwrist_y = [kps[i][10,1] if kps[i][10,2] == 1 else -1 for i in range(total_frames)]
    pt5_leftshoulder_x = [kps[i][5,0] if kps[i][5,2] == 1 else -1 for i in range(total_frames)]
    pt5_leftshoulder_y = [kps[i][5,1] if kps[i][5,2] == 1 else -1 for i in range(total_frames)]
    pt6_leftelbow_x = [kps[i][7,0] if kps[i][7,2] == 1 else -1 for i in range(total_frames)]
    pt6_leftelbow_y = [kps[i][7,1] if kps[i][7,2] == 1 else -1 for i in range(total_frames)]
    pt7_leftwrist_x = [kps[i][9,0] if kps[i][9,2] == 1 else -1 for i in range(total_frames)]
    pt7_leftwrist_y = [kps[i][9,1] if kps[i][9,2] == 1 else -1 for i in range(total_frames)]
    pt8_navel_x = [-1 for i in range(total_frames)]
    pt8_navel_y = [-1 for i in range(total_frames)]
    pt9_righthip_x = [kps[i][12,0] if kps[i][12,2] == 1 else -1 for i in range(total_frames)]
    pt9_righthip_y = [kps[i][12,1] if kps[i][12,2] == 1 else -1 for i in range(total_frames)]
    pt10_rightknee_x = [kps[i][14,0] if kps[i][14,2] == 1 else -1 for i in range(total_frames)]
    pt10_rightknee_y = [kps[i][14,1] if kps[i][14,2] == 1 else -1 for i in range(total_frames)]
    pt11_rightankle_x = [kps[i][16,0] if kps[i][16,2] == 1 else -1 for i in range(total_frames)]
    pt11_rightankle_y= [kps[i][16,1] if kps[i][16,2] == 1 else -1 for i in range(total_frames)]
    pt12_lefthip_x = [kps[i][11,0] if kps[i][11,2] == 1 else -1 for i in range(total_frames)]
    pt12_lefthip_y = [kps[i][11,1] if kps[i][11,2] == 1 else -1 for i in range(total_frames)]
    pt13_leftknee_x = [kps[i][13,0] if kps[i][13,2] == 1 else -1 for i in range(total_frames)]
    pt13_leftknee_y = [kps[i][13,1] if kps[i][13,2] == 1 else -1 for i in range(total_frames)]
    pt14_leftankle_x = [kps[i][15,0] if kps[i][15,2] == 1 else -1 for i in range(total_frames)]
    pt14_leftankle_y = [kps[i][15,1] if kps[i][15,2] == 1 else -1 for i in range(total_frames)]
    
    df_pose = pd.DataFrame(OrderedDict({'video_file_name':video_file_name, 'frame_num':frame_num, 'model_name':model_name_s,'model_version':model_version_s, 'confidence':confidence, 'pt0_forehead_x':pt0_forehead_x,'pt0_forehead_y':pt0_forehead_y, 'pt1_upperchest_x':pt1_upperchest_x, 'pt1_upperchest_y':pt1_upperchest_y, 'pt2_rightshoulder_x':pt2_rightshoulder_x, 'pt2_rightshoulder_y':pt2_rightshoulder_y, 'pt3_rightelbow_x':pt3_rightelbow_x, 'pt3_rightelbow_y':pt3_rightelbow_y,'pt4_rightwrist_x':pt4_rightwrist_x, 'pt4_rightwrist_y':pt4_rightwrist_y, 'pt5_leftshoulder_x':pt5_leftshoulder_x, 'pt5_leftshoulder_y':pt5_leftshoulder_y, 'pt6_leftelbow_x':pt6_leftelbow_x,'pt6_leftelbow_y':pt6_leftelbow_y, 'pt7_leftwrist_x':pt7_leftwrist_x, 'pt7_leftwrist_y':pt7_leftwrist_y, 'pt8_navel_x':pt8_navel_x, 'pt8_navel_y':pt8_navel_y, 'pt9_righthip_x':pt9_righthip_x, 'pt9_righthip_y':pt9_righthip_y, 'pt10_rightknee_x':pt10_rightknee_x, 'pt10_rightknee_y':pt10_rightknee_y, 'pt11_rightankle_x':pt11_rightankle_x, 'pt11_rightankle_y':pt11_rightankle_y, 'pt12_lefthip_x':pt12_lefthip_x, 'pt12_lefthip_y':pt12_lefthip_y, 'pt13_leftknee_x':pt13_leftknee_x, 'pt13_leftknee_y':pt13_leftknee_y, 'pt14_leftankle_x':pt14_leftankle_x, 'pt14_leftankle_y':pt14_leftankle_y}))
    
    #print (df_pose.columns)
    return df_pose