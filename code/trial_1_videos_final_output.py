#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import pandas as pd
import cv2
import os

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Source for code: https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

# Specify the paths for the 2 files
protoFile = "../inputs_outputs/models/pose_deploy.prototxt"
weightsFile = "../inputs_outputs/models/pose_iter_584000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


# In[3]:


# Provide the video filepath as a string below
# For example: "my_video.mp4" or "/content/drive/My Drive/my_folder/my_video.mp4"

my_video = input("Provide the video filepath as a string.\n\nINPUT-> ")
print("\n")


# In[4]:


orient = input("Is the person in the video facing towards left or right of the video frame?\n\nINPUT-> ")
orient = orient.lower()

print("\nPROCESSING... PLEASE WAIT!\n")


# In[5]:


# %%time

# Reading the video from filepath provided above and passing it through the age clasification method defined above.
# Source 1: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# Source 2: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# Creating a VideoCapture object.
cap = cv2.VideoCapture(my_video)

# Checking if video can be accessed successfully.
if (cap.isOpened() == False): 
    print("Unable to read video!")

total_frames = 0

while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        total_frames += 1
        
        # Exiting if "Q" key is pressed on the keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Releasing the VideoCapture object.
cap.release()

# print(total_frames)


# In[6]:


def find_skeleton(frame, frame_counter):

    frame_copy = np.copy(frame)

    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368

    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame_copy, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []
    for i in range(15):
        # Confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frame_copy.shape[1] * point[0]) / W
        y = (frame_copy.shape[0] * point[1]) / H

        if prob > 0.5:
            cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame_copy, f"{i}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame_copy, f"frame = {frame_counter}", (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))

        else:
            points.append(None)
    
    return frame_copy, points

    # cv2.imshow("Output-Keypoints",frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# In[7]:


# Defining a function to return the video filepath with a new filename.
# If INPUT filepath is "my_folder1/my_folder2/my_video.mp4", OUTPUT filepath will be "my_folder1/my_folder2/my_video_WITH_AGE.mp4"

def new_vid_name(org_vid_path):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    new_vid_name_ext = vid_name+"_WITH_BODY_POINTS"+".mp4"
    new_vid_path = os.path.join(vid_path, new_vid_name_ext)

    return new_vid_path


# In[8]:


# %%time

# Reading the video from filepath provided above and passing it through the age clasification method defined above.
# Source 1: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# Source 2: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# Creating a VideoCapture object.
cap = cv2.VideoCapture(my_video)

# Checking if video can be accessed successfully.
if (cap.isOpened() == False): 
    print("Unable to read video!")

# Getting the video frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Defining the codec and creating a VideoWriter object to save the output video at the same location.
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# new_my_video = new_vid_name(my_video)
# out = cv2.VideoWriter(new_my_video, fourcc, 18, (frame_width, frame_height))

# Defining a new dataframe to store the skeleton point coordinates from each frame of the video.
video_points_df = pd.DataFrame(columns=list(range(1,26)))
frame_counter = 1

while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        print(f"Analysing video - frame {frame_counter} of {total_frames}...")
        
        # Running human skeleton points detection on the grabbed frame.
        skeleton_frame, skeleton_frame_points = find_skeleton(frame, frame_counter)
        
        # Saving frame to output video using the VideoWriter object defined above.
        # out.write(skeleton_frame)
        
        # Displaying the frame with age detected.
        # cv2.imshow("Output Video", skeleton_frame)
        
        # Saving frame output skeleton point coordinates as a new row in above defined dataframe.
        video_points_df[frame_counter] = skeleton_frame_points
        frame_counter += 1
        
        # Exiting if "Q" key is pressed on the keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Releasing the VideoCapture and VideoWriter objects, and closing the displayed frame.
cap.release()
# out.release()
# cv2.destroyAllWindows()
# print(f"Saved to {new_my_video}")


# In[9]:


# frame_counter


# In[10]:


# video_points_df


# ## Initial Analysis

# In[11]:


df = video_points_df.T.copy()


# In[12]:


# df.head(40)


# In[13]:


# frame_width, frame_height


# ## Data Imputation

# In[14]:


vjump_points = [0, 1, 8]

if orient == "left":
    vjump_points.extend([5, 6, 7, 12, 13, 14])
elif orient == "right":
    vjump_points.extend([2, 3, 4, 9, 10, 11])

# sorted(vjump_points)


# In[15]:


df_imputed = df[vjump_points].copy()
# df_imputed.head(40)


# In[16]:


def find_break_in_series(points_series, start_i):
    
    break_start_i = start_i - 1
    break_end_i = start_i
    break_found = False
    continue_crawling = True
    
    while (continue_crawling == True) and (break_end_i < points_series.index[-1]):
    
        if points_series[break_end_i] != None:
            break_found = False
            break_start_i = break_end_i
            break_end_i += 1
            
        else:
            break_found = True
            # break_end_i += 1
            
            while (break_found == True) and (break_end_i <= points_series.index[-1]):
                
                if points_series[break_end_i] == None:
                    break_end_i += 1
                else:
                    break_found = False
                    
            continue_crawling = False
        
    return break_start_i, break_end_i

def impute_break_in_series(points_series, start_i, end_i):
    
    diff = end_i - start_i
    start_x, start_y = points_series[start_i]
    end_x, end_y = points_series[end_i]
    
    diff_x = (end_x - start_x) / diff
    diff_y = (end_y - start_y) / diff
    
    multiplier = 1
    
    for i in range(start_i+1, end_i):
        x = int(round(start_x + (diff_x * multiplier)))
        y = int(round(start_y + (diff_y * multiplier)))
        points_series[i] = (x, y)
        multiplier += 1

def imputing_missing_points(points_series):
    
    series_start_i = points_series.index[0]
    series_end_i = points_series.index[-1]
    
    current_i = series_start_i
    
    while current_i < series_end_i:
        
        if points_series[current_i] == None:
            current_i += 1
            
        else:
            break_start_i, break_end_i = find_break_in_series(points_series, current_i)
            
            if break_end_i - break_start_i == 1:
                break
            elif break_end_i > series_end_i:
                break
            else: 
                impute_break_in_series(points_series, break_start_i, break_end_i)
                current_i = break_end_i
    
    return points_series


# In[17]:


print()
print("Continuing analysis - data imputation...\n")

for i in vjump_points:
    df_imputed[i] = imputing_missing_points(df_imputed[i])


# In[18]:


# df_imputed.head(40)


# In[19]:


def overlay_imputed_skeleton(frame, points_row_imputed, points_row_original, frame_counter):

    frame_copy = np.copy(frame)

    for i in points_row_imputed.index:
        
        if points_row_imputed[i] != None:
            
            if points_row_original[i] != None:
        
                cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            
            else:
                
                cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                
        cv2.putText(frame_copy, f"frame = {frame_counter}", (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                
    return frame_copy


# ## Data Points Normalization

# In[20]:


df_imputed_norm = df_imputed.copy()


# In[21]:


# df_imputed_norm.head(40)


# In[22]:


origin_point = 0


# In[23]:


def find_upper_body_length():
    
    upper_body_length = 0
    
    for i in df.index:
        
        if ((df.loc[i, 1] != None) and (df.loc[i, 8] != None)):
            
            upper_chest_x, upper_chest_y = df.loc[i, 1]
            navel_x, navel_y = df.loc[i, 8]
            
            upper_body_length = int(round(np.sqrt((upper_chest_x - navel_x)**2 + (upper_chest_y - navel_y)**2)))
            
            break
            
    # print(upper_body_length)
    return upper_body_length


# In[24]:


def normalizing_body_points(row):
    
    # max_body_dim_x, max_body_dim_y = find_max_body_dimensions()
    upper_body_length = find_upper_body_length()
    
    if row[origin_point] == None:
        return row
    
    origin_x, origin_y = row[origin_point]
    
    for i in row.index:
        
        if i == origin_point:
            norm_x = 0.0
            norm_y = 0.0
            row[i] = (norm_x, norm_y)
            
        elif row[i] == None:
            pass
        
        else:
            i_x, i_y = row[i]
            norm_x = (i_x - origin_x) / (upper_body_length * 3)
            norm_y = (i_y - origin_y) / (upper_body_length * 3)
            norm_y = -1 * norm_y
            row[i] = (norm_x, norm_y)
    
    return row


# In[25]:


print("Continuing analysis - data normalization...\n")

for i in df_imputed_norm.index:
    df_imputed_norm.loc[i] = normalizing_body_points(df_imputed_norm.loc[i])


# In[26]:


# df_imputed_norm.head(40)


# In[27]:


# df_imputed_norm.tail(40)


# ## Setting Rules for Grading Vertical Jump

# In[28]:


# df_imputed.head(40)


# In[29]:


# Tracking body point 0 in Y-axis to get frame number of peak in vertical jump.

print("Continuing analysis - identifying critical frames...\n")

forehead_y_movement = [frame_height - x[1] for x in df_imputed[0]]


# In[30]:


jump_peak_indexes = []

for i, y in enumerate(forehead_y_movement):
    if y == max(forehead_y_movement):
        jump_peak_indexes.append(i+1)


# In[31]:


jump_peak_frame = int(np.median(jump_peak_indexes))
# jump_peak_frame


# In[32]:


jump_squat_indexes = []

for i, y in enumerate(forehead_y_movement[:jump_peak_frame]):
    if y == min(forehead_y_movement[:jump_peak_frame]):
        jump_squat_indexes.append(i+1)


# In[33]:


jump_squat_frame = int(np.median(jump_squat_indexes))
# jump_squat_frame


# In[34]:


jump_land_indexes = []

for i, y in enumerate(forehead_y_movement[jump_peak_frame:]):
    if y == min(forehead_y_movement[jump_peak_frame:]):
        jump_land_indexes.append(i+1+jump_peak_frame)


# In[35]:


jump_land_frame = int(np.median(jump_land_indexes))
# jump_land_frame


# In[36]:


def calculate_angle_between_three_points(point_1, point_2, point_3):
    
    x_1, y_1 = point_1
    x_2, y_2 = point_2
    x_3, y_3 = point_3
    
    side_1_2 = np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
    side_2_3 = np.sqrt((x_3 - x_2)**2 + (y_3 - y_2)**2)
    side_3_1 = np.sqrt((x_1 - x_3)**2 + (y_1 - y_3)**2)
    
    cos_theta_rad = ((side_1_2**2 + side_2_3**2 - side_3_1**2) / (2 * side_1_2 * side_2_3))
    
    theta_rad = np.arccos(cos_theta_rad)
    
    theta_deg = np.rad2deg(theta_rad)
    
    return theta_deg


# ## FINAL OUTPUT

# In[37]:


def jump_quality_at_squat_frame(frame, points_row_original, points_row_imputed, points_row_norm):
    
    frame_copy = np.copy(frame)
    
    temp_arm_points = []
    temp_leg_points = []
    temp_torso_points = []
    
    if orient == "left":
        temp_arm_points = [5, 6, 7]
        temp_leg_points = [12, 13, 14]
        temp_torso_points = [5, 12]
    elif orient == "right":
        temp_arm_points = [2, 3, 4]
        temp_leg_points = [9, 10, 11]
        temp_torso_points = [2, 9]
        
    elbow_angle = calculate_angle_between_three_points(points_row_norm[temp_arm_points[0]],
                                                       points_row_norm[temp_arm_points[1]],
                                                       points_row_norm[temp_arm_points[2]]
                                                      )
    
    knee_angle = calculate_angle_between_three_points(points_row_norm[temp_leg_points[0]],
                                                      points_row_norm[temp_leg_points[1]],
                                                      points_row_norm[temp_leg_points[2]]
                                                     )
    
    shoulder_angle = calculate_angle_between_three_points(points_row_norm[temp_arm_points[1]],
                                                          points_row_norm[temp_arm_points[0]],
                                                          points_row_norm[temp_torso_points[1]]
                                                         )
    
    hip_angle = calculate_angle_between_three_points(points_row_norm[temp_torso_points[0]],
                                                     points_row_norm[temp_torso_points[1]],
                                                     points_row_norm[temp_leg_points[1]]
                                                    )
    
    correct_squat_posture = False
    elbow_angle_pass = False
    knee_angle_pass = False
    shoulder_angle_pass = False
    hip_angle_pass = False
    
    for i, p in enumerate(temp_arm_points[:-1]):
        if elbow_angle > 160:
            elbow_angle_pass = True
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_arm_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_arm_points[i+1]], (0, 0, 255), thickness=2)
    
    for i, p in enumerate(temp_leg_points[:-1]):
        if 75 < knee_angle < 105:
            knee_angle_pass = True
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 0, 255), thickness=2)
            
    for i in points_row_imputed.index:
        if points_row_imputed[i] != None:
            if points_row_original[i] != None:
                cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame_copy, f"frame = {frame_counter}", (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    
    
    if elbow_angle_pass and knee_angle_pass:
        correct_squat_posture = True
    
    frame_text = ""
    if correct_squat_posture:
        frame_text = "Squat posture : PASS"
    else:
        frame_text = "Squat posture : FAIL"
    
    report_text = f'''
    {frame_text} (frame {frame_counter} of {total_frames})
    Body angles:
    \tElbow:\t{round(elbow_angle, 2)} deg\t[correct range: >160 deg]
    \tKnee:\t{round(knee_angle, 2)} deg\t[correct range: 75-105 deg]
    \tShldr:\t{round(shoulder_angle, 2)} deg
    \tHip:\t{round(knee_angle, 2)} deg
    '''

    return frame_copy, frame_text, report_text


# In[38]:


def jump_quality_at_peak_frame(frame, points_row_original, points_row_imputed, points_row_norm):
    
    frame_copy = np.copy(frame)
    
    temp_arm_points = []
    temp_leg_points = []
    temp_torso_points = []
    
    if orient == "left":
        temp_arm_points = [5, 6, 7]
        temp_leg_points = [12, 13, 14]
        temp_torso_points = [5, 12]
    elif orient == "right":
        temp_arm_points = [2, 3, 4]
        temp_leg_points = [9, 10, 11]
        temp_torso_points = [2, 9]
        
    elbow_angle = calculate_angle_between_three_points(points_row_norm[temp_arm_points[0]],
                                                       points_row_norm[temp_arm_points[1]],
                                                       points_row_norm[temp_arm_points[2]]
                                                      )
    
    knee_angle = calculate_angle_between_three_points(points_row_norm[temp_leg_points[0]],
                                                      points_row_norm[temp_leg_points[1]],
                                                      points_row_norm[temp_leg_points[2]]
                                                      )
    
    torso_angle = calculate_angle_between_three_points((points_row_norm[temp_torso_points[0]][0], points_row_norm[temp_torso_points[0]][1] - 1), 
                                                       points_row_norm[temp_torso_points[0]],
                                                       points_row_norm[temp_torso_points[1]]
                                                      )
    
    correct_peak_posture = False
    elbow_angle_pass = False
    knee_angle_pass = False
    torso_angle_pass = False
    
    for i, p in enumerate(temp_arm_points[:-1]):
        if elbow_angle > 150:
            elbow_angle_pass = True
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_arm_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_arm_points[i+1]], (0, 0, 255), thickness=2)
    
    for i, p in enumerate(temp_leg_points[:-1]):
        if knee_angle > 150:
            knee_angle_pass = True
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 0, 255), thickness=2)
            
    for i, p in enumerate(temp_torso_points[:-1]):
        if torso_angle < 20:
            torso_angle_pass = True
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_torso_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_torso_points[i+1]], (0, 0, 255), thickness=2)
    
    for i in points_row_imputed.index:
        if points_row_imputed[i] != None:
            if points_row_original[i] != None:
                cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame_copy, f"frame = {frame_counter}", (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    
    
    if elbow_angle_pass and knee_angle_pass and torso_angle_pass:
        correct_peak_posture = True
    
    frame_text = ""
    if correct_peak_posture:
        frame_text = "Jump Peak posture : PASS"
    else:
        frame_text = "Jump Peak posture : FAIL"
    
    report_text = f'''
    {frame_text} (frame {frame_counter} of {total_frames})
    Body angles:
    \tElbow:\t{round(elbow_angle, 2)} deg\t[correct range: >150 deg]
    \tKnee:\t{round(knee_angle, 2)} deg\t[correct range: >150 deg]
    \tTorso:\t{round(torso_angle, 2)} deg\t[correct range: <20 deg]
    '''

    return frame_copy, frame_text, report_text


# In[39]:


def calculate_distance_from_squat_to_land():
    
    upper_body_length = find_upper_body_length()
    
    ankle_point = 0
    
    if orient == "left":
        ankle_point = 14
    elif orient == "right":
        ankle_point = 11
        
    squat_ankle_x, squat_ankle_y = df_imputed.loc[jump_squat_frame, ankle_point]
    land_ankle_x, land_ankle_y = df_imputed.loc[jump_land_frame, ankle_point]
    
    landing_distance = np.sqrt((land_ankle_x - squat_ankle_x)**2 + (land_ankle_y - squat_ankle_y)**2)
    
    return landing_distance / upper_body_length


# In[40]:


def jump_quality_at_land_frame(frame, points_row_original, points_row_imputed, points_row_norm):
    
    frame_copy = np.copy(frame)
    
    temp_arm_points = []
    temp_leg_points = []
    temp_torso_points = []
    
    if orient == "left":
        temp_arm_points = [5, 6, 7]
        temp_leg_points = [12, 13, 14]
        temp_torso_points = [5, 12]
    elif orient == "right":
        temp_arm_points = [2, 3, 4]
        temp_leg_points = [9, 10, 11]
        temp_torso_points = [2, 9]
    
    knee_angle = calculate_angle_between_three_points(points_row_norm[temp_leg_points[0]],
                                                      points_row_norm[temp_leg_points[1]],
                                                      points_row_norm[temp_leg_points[2]]
                                                     )
    
    torso_angle = calculate_angle_between_three_points((points_row_norm[temp_torso_points[0]][0], points_row_norm[temp_torso_points[0]][1] - 1), 
                                                       points_row_norm[temp_torso_points[0]],
                                                       points_row_norm[temp_torso_points[1]]
                                                      )
    
    correct_land_posture = False
    knee_angle_pass = False
    torso_angle_pass = False
    land_distance_pass = False
    
    for i, p in enumerate(temp_leg_points[:-1]):
        if knee_angle < 150:
            knee_angle_pass = True
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 0, 255), thickness=2)
            
    for i, p in enumerate(temp_torso_points[:-1]):
        if torso_angle < 10:
            torso_angle_pass = True
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_torso_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_torso_points[i+1]], (0, 0, 255), thickness=2)
    
    for i in points_row_imputed.index:
        if points_row_imputed[i] != None:
            if points_row_original[i] != None:
                cv2.circle(frame_copy, points_row_imputed[i], 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame_copy, points_row_imputed[i], 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                # cv2.putText(frame_copy, f"{i}", points_row_imputed[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(frame_copy, f"frame = {frame_counter}", (10, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    
    
    land_distance = calculate_distance_from_squat_to_land()
    if land_distance <= 1:
        land_distance_pass = True
    
    if knee_angle_pass and torso_angle_pass and land_distance_pass:
        correct_land_posture = True
    
    frame_text = ""
    if correct_land_posture:
        frame_text = "Landing posture : PASS"
    else:
        frame_text = "Landing posture : FAIL"
    
    report_text = f'''
    {frame_text} (frame {frame_counter} of {total_frames})
    Body angles:
    \tKnee:\t{round(knee_angle, 2)} deg\t[correct range: <150 deg]
    \tTorso:\t{round(torso_angle, 2)} deg\t[correct range: <10 deg]
    
    Landing distance: {round(land_distance , 2)} X Upper Body length [correct range: <=1]
    '''

    return frame_copy, frame_text, report_text


# In[41]:


def overlay_frame_text(frame, squat_frame_text, peak_frame_text, land_frame_text):
    
    frame_copy = np.copy(frame)
    
    if "PASS" in squat_frame_text:
        cv2.putText(frame_copy, squat_frame_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    elif "FAIL" in squat_frame_text:
        cv2.putText(frame_copy, squat_frame_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    else:
        cv2.putText(frame_copy, "Squat posture : ANALYSING...", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)
    
    if "PASS" in peak_frame_text:
        cv2.putText(frame_copy, peak_frame_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    elif "FAIL" in peak_frame_text:
        cv2.putText(frame_copy, peak_frame_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    else:
        cv2.putText(frame_copy, "Jump Peak posture : ANALYSING...", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)
    
    if "PASS" in land_frame_text:
        cv2.putText(frame_copy, land_frame_text, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    elif "FAIL" in land_frame_text:
        cv2.putText(frame_copy, land_frame_text, (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    else:
        cv2.putText(frame_copy, "Landing posture : ANALYSING...", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, lineType=cv2.LINE_AA)
        
    return frame_copy


# In[42]:


# Defining a function to return the video filepath with a new filename.
# If INPUT filepath is "my_folder1/my_folder2/my_video.mp4", OUTPUT filepath will be "my_folder1/my_folder2/my_video_WITH_AGE.mp4"

def new_final_vid_name(org_vid_path):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    new_vid_name_ext = vid_name+"_ANALYSIS_VIDEO"+".mp4"
    new_vid_path = os.path.join(vid_path, new_vid_name_ext)

    return new_vid_path


# In[43]:


# Defining a function to return the video filepath with a new filename.
# If INPUT filepath is "my_folder1/my_folder2/my_video.mp4", OUTPUT filepath will be "my_folder1/my_folder2/my_video_WITH_AGE.mp4"

def new_final_report_name(org_vid_path):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    report_name_ext = vid_name+"_ANALYSIS_REPORT"+".txt"
    report_path = os.path.join(vid_path, report_name_ext)

    return report_path


# In[47]:


# %%time

# Reading the video from filepath provided above and passing it through the age clasification method defined above.
# Source 1: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# Source 2: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# Creating a VideoCapture object.
cap = cv2.VideoCapture(my_video)

# Checking if video can be accessed successfully.
if (cap.isOpened() == False): 
    print("Unable to read video!")

# Defining the codec and creating a VideoWriter object to save the output video at the same location.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_my_video = new_final_vid_name(my_video)
out = cv2.VideoWriter(new_my_video, fourcc, 18, (frame_width, frame_height))
    
# Defining a new dataframe to store the skeleton point coordinates from each frame of the video.
frame_counter = 1
squat_frame_text = ""
peak_frame_text = ""
land_frame_text = ""

final_report_text = f'''VERTICAL JUMP REPORT
Video Path : {my_video}
'''

while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        if frame_counter == jump_squat_frame:
            skeleton_frame, squat_frame_text, report_text = jump_quality_at_squat_frame(frame, df.loc[frame_counter], df_imputed.loc[frame_counter], df_imputed_norm.loc[frame_counter])
            final_report_text += report_text
            
        elif frame_counter == jump_peak_frame:
            skeleton_frame, peak_frame_text, report_text = jump_quality_at_peak_frame(frame, df.loc[frame_counter], df_imputed.loc[frame_counter], df_imputed_norm.loc[frame_counter])
            final_report_text += report_text
            
        elif frame_counter == jump_land_frame:
            skeleton_frame, land_frame_text, report_text = jump_quality_at_land_frame(frame, df.loc[frame_counter], df_imputed.loc[frame_counter], df_imputed_norm.loc[frame_counter])
            final_report_text += report_text
            
        else:
            skeleton_frame = overlay_imputed_skeleton(frame, df_imputed.loc[frame_counter], df.loc[frame_counter], frame_counter)
            
        skeleton_frame = overlay_frame_text(skeleton_frame, squat_frame_text, peak_frame_text, land_frame_text)
        
        # Saving frame to output video using the VideoWriter object defined above.
        out.write(skeleton_frame)
        
        # Displaying the frame with age detected.
        # cv2.imshow("Output Video", skeleton_frame)
        
        frame_counter += 1
        
        # Exiting if "Q" key is pressed on the keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Releasing the VideoCapture and VideoWriter objects, and closing the displayed frame.
cap.release()
out.release()
# cv2.destroyAllWindows()
print(f"Analysis video saved at {new_my_video}")


# In[45]:


final_report = new_final_report_name(my_video)

with open(final_report, "w") as text_file:
    text_file.write(final_report_text)


# In[46]:


print()

print(final_report_text)


# In[ ]:




