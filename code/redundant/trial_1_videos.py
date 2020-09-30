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


# In[4]:


# Defining a function to return the video filepath with a new filename.
# If INPUT filepath is "my_folder1/my_folder2/my_video.mp4", OUTPUT filepath will be "my_folder1/my_folder2/my_video_WITH_AGE.mp4"

def new_vid_name(org_vid_path):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    new_vid_name_ext = vid_name+"_WITH_BODY_POINTS"+".mp4"
    new_vid_path = os.path.join(vid_path, new_vid_name_ext)

    return new_vid_path


# In[5]:


# Provide the video filepath as a string below
# For example: "my_video.mp4" or "/content/drive/My Drive/my_folder/my_video.mp4"

# my_video = "../inputs_outputs/videos/vertical_jump_side_trial_1.mp4"
my_video = input("Provide the video filepath as a string.\n\n")
print("\n")


# In[14]:


# orient = "left"
orient = input("Is the person in the video facing towards left or right of the video frame?\n\n")
orient = orient.lower()


print("\n\nPROCESSING... PLEASE WAIT!\n\n\n")


# In[6]:


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
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_my_video = new_vid_name(my_video)
out = cv2.VideoWriter(new_my_video, fourcc, 18, (frame_width, frame_height))

# Defining a new dataframe to store the skeleton point coordinates from each frame of the video.
video_points_df = pd.DataFrame(columns=list(range(1,26)))
frame_counter = 1

while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        # Running human skeleton points detection on the grabbed frame.
        skeleton_frame, skeleton_frame_points = find_skeleton(frame, frame_counter)
        
        # Saving frame to output video using the VideoWriter object defined above.
        out.write(skeleton_frame)
        
        # Displaying the frame with age detected.
        cv2.imshow("Output Video", skeleton_frame)
        
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
out.release()
cv2.destroyAllWindows()
print(f"Saved to {new_my_video}")


# ## Initial Analysis

# In[9]:


df = video_points_df.T.copy()


# ## Data Imputation


# In[15]:


vjump_points = [0, 1, 8]

if orient == "left":
    vjump_points.extend([5, 6, 7, 12, 13, 14])
elif orient == "right":
    vjump_points.extend([2, 3, 4, 9, 10, 11])

# sorted(vjump_points)


# In[16]:


df_imputed = df[vjump_points].copy()
# df_imputed.head(40)


# In[17]:


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


# In[18]:


for i in vjump_points:
    df_imputed[i] = imputing_missing_points(df_imputed[i])


# In[22]:


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


# In[23]:


# Defining a function to return the video filepath with a new filename.
# If INPUT filepath is "my_folder1/my_folder2/my_video.mp4", OUTPUT filepath will be "my_folder1/my_folder2/my_video_WITH_AGE.mp4"

def new_imputed_vid_name(org_vid_path):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    new_vid_name_ext = vid_name+"_WITH_IMPUTED_BODY_POINTS"+".mp4"
    new_vid_path = os.path.join(vid_path, new_vid_name_ext)

    return new_vid_path


# In[24]:


# Reading the video from filepath provided above and passing it through the age clasification method defined above.
# Source 1: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# Source 2: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# Creating a VideoCapture object.
cap = cv2.VideoCapture(my_video)

# Checking if video can be accessed successfully.
if (cap.isOpened() == False): 
    print("Unable to read video!")

# Getting the video frame width and height.
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# Defining the codec and creating a VideoWriter object to save the output video at the same location.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_my_video = new_imputed_vid_name(my_video)
out = cv2.VideoWriter(new_my_video, fourcc, 18, (frame_width, frame_height))

# Defining a new dataframe to store the skeleton point coordinates from each frame of the video.
frame_counter = 1

while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        # Running human skeleton points detection on the grabbed frame.
        skeleton_frame = overlay_imputed_skeleton(frame, df_imputed.loc[frame_counter], df.loc[frame_counter], frame_counter)
        
        # Saving frame to output video using the VideoWriter object defined above.
        out.write(skeleton_frame)
        
        # Displaying the frame with age detected.
        cv2.imshow("Output Video", skeleton_frame)
        
        # Saving frame output skeleton point coordinates as a new row in above defined dataframe.
        # video_points_df[frame_counter] = skeleton_frame_points
        frame_counter += 1
        
        # Exiting if "Q" key is pressed on the keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Releasing the VideoCapture and VideoWriter objects, and closing the displayed frame.
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved to {new_my_video}")


# ## Data Points Normalization

# In[25]:


df_imputed_norm = df_imputed.copy()


# In[27]:


origin_point = 0


# In[28]:


def find_max_body_dimensions():
    
    max_body_dim_x = 0
    max_body_dim_y = 0
    
    for i in df.index:
        
        if df.loc[i, origin_point] != None:
        
            origin_x, origin_y = df.loc[i, origin_point]

            for p in [point for point in vjump_points if point != origin_point]:

                if df.loc[i, p] != None:

                    p_x, p_y = df.loc[i, p]

                    diff_x = abs(p_x - origin_x)
                    diff_y = abs(p_y - origin_y)

                    if diff_x > max_body_dim_x:
                        max_body_dim_x = diff_x
                    if diff_y > max_body_dim_y:
                        max_body_dim_y = diff_y
    
    return max_body_dim_x, max_body_dim_y

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


# In[29]:


for i in df_imputed_norm.index:
    df_imputed_norm.loc[i] = normalizing_body_points(df_imputed_norm.loc[i])


# In[34]:


def normalized_skeleton(points_row_norm, points_row_original, frame_counter):

    frame = cv2.imread("../inputs_outputs/white_frame.jpg")

    for i in points_row_norm.index:
        
        if points_row_norm[i] != None:
            
            x_norm, y_norm = points_row_norm[i]
            x_norm_sc = int(round(x_norm * 1000 + 1000))
            y_norm_sc = int(round(y_norm * 1000 + 1000))
            
            if points_row_original[i] != None:
        
                cv2.circle(frame, (x_norm_sc, y_norm_sc), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, f"{i}", (x_norm_sc, y_norm_sc), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            
            else:
                
                cv2.circle(frame, (x_norm_sc, y_norm_sc), 5, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, f"{i}", (x_norm_sc, y_norm_sc), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            
        cv2.putText(frame, f"frame = {frame_counter}", (10, 1990), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, lineType=cv2.LINE_AA)
                
    return frame


# In[35]:


# Defining a function to return the video filepath with a new filename.
# If INPUT filepath is "my_folder1/my_folder2/my_video.mp4", OUTPUT filepath will be "my_folder1/my_folder2/my_video_WITH_AGE.mp4"

def new_normalized_vid_name(org_vid_path):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    new_vid_name_ext = vid_name+"_WITH_NORMALIZED_BODY_POINTS"+".mp4"
    new_vid_path = os.path.join(vid_path, new_vid_name_ext)

    return new_vid_path


# In[36]:


# Reading the video from filepath provided above and passing it through the age clasification method defined above.
# Source 1: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# Source 2: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# Creating a VideoCapture object.
cap = cv2.VideoCapture(my_video)

# Checking if video can be accessed successfully.
if (cap.isOpened() == False): 
    print("Unable to read video!")

# Getting the video frame width and height.
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# Defining the codec and creating a VideoWriter object to save the output video at the same location.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_my_video = new_normalized_vid_name(my_video)
out = cv2.VideoWriter(new_my_video, fourcc, 18, (2000, 2000))

# Defining a new dataframe to store the skeleton point coordinates from each frame of the video.
frame_counter = 1

while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        # Running human skeleton points detection on the grabbed frame.
        skeleton_frame = normalized_skeleton(df_imputed_norm.loc[frame_counter], df.loc[frame_counter], frame_counter)
        
        # Saving frame to output video using the VideoWriter object defined above.
        out.write(skeleton_frame)
        
        # Displaying the frame with age detected.
        cv2.imshow("Output Video", skeleton_frame)
        
        # Saving frame output skeleton point coordinates as a new row in above defined dataframe.
        # video_points_df[frame_counter] = skeleton_frame_points
        frame_counter += 1
        
        # Exiting if "Q" key is pressed on the keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Releasing the VideoCapture and VideoWriter objects, and closing the displayed frame.
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved to {new_my_video}")


# In[41]:


# Tracking body point 0 in Y-axis to get frame number of peak in vertical jump.

forehead_y_movement = [frame_height - x[1] for x in df_imputed[0]]


# In[42]:


jump_peak_indexes = []

for i, y in enumerate(forehead_y_movement):
    if y == max(forehead_y_movement):
        jump_peak_indexes.append(i+1)


# In[43]:


jump_peak_frame = int(np.median(jump_peak_indexes))
# jump_peak_frame


# In[44]:


jump_squat_indexes = []

for i, y in enumerate(forehead_y_movement[:jump_peak_frame]):
    if y == min(forehead_y_movement[:jump_peak_frame]):
        jump_squat_indexes.append(i+1)


# In[45]:


jump_squat_frame = int(np.median(jump_squat_indexes))
# jump_squat_frame


# In[46]:


jump_land_indexes = []

for i, y in enumerate(forehead_y_movement[jump_peak_frame:]):
    if y == min(forehead_y_movement[jump_peak_frame:]):
        jump_land_indexes.append(i+1+jump_peak_frame)


# In[47]:


jump_land_frame = int(np.median(jump_land_indexes))
# jump_land_frame


# In[50]:


# Defining a function to return the image filepath with a new filename.
# If INPUT filepath is "my_folder1/my_folder2/my_image.jpg", OUTPUT filepath will be "my_folder1/my_folder2/my_image_WITH_AGE.jpg"

def new_frame_name(org_vid_path, frame_counter):
    vid_path, vid_name_ext = os.path.split(org_vid_path)
    vid_name, vid_ext = os.path.splitext(vid_name_ext)

    new_frame_name_ext = ""
    if frame_counter == jump_squat_frame:
        new_frame_name_ext = vid_name+"_1_JUMP_SQUAT"+".png"
    elif frame_counter == jump_peak_frame:
        new_frame_name_ext = vid_name+"_2_JUMP_PEAK"+".png"
    elif frame_counter == jump_land_frame:
        new_frame_name_ext = vid_name+"_3_JUMP_LAND"+".png"
        
    new_frame_path = os.path.join(vid_path, new_frame_name_ext)

    return new_frame_path


# In[49]:


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
    
    arm_points_x = []
    for p in temp_arm_points:
        arm_points_x.append(points_row_norm[p][0])
    arm_straightness = max(arm_points_x) - min(arm_points_x)
    
    leg_points_x = []
    for p in temp_leg_points:
        leg_points_x.append(points_row_norm[p][0])
    leg_straightness = max(leg_points_x) - min(leg_points_x)
    
    torso_points_x = []
    for p in temp_torso_points:
        torso_points_x.append(points_row_norm[p][0])
    torso_straightness = max(torso_points_x) - min(torso_points_x)
    
    for i, p in enumerate(temp_arm_points[:-1]):
        if arm_straightness <= 0.05:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_arm_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_arm_points[i+1]], (0, 0, 255), thickness=2)
    
    for i, p in enumerate(temp_leg_points[:-1]):
        if leg_straightness <= 0.05:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 255, 0), thickness=2)
        else:
            cv2.line(frame_copy, points_row_imputed[p], points_row_imputed[temp_leg_points[i+1]], (0, 0, 255), thickness=2)
            
    for i, p in enumerate(temp_torso_points[:-1]):
        if torso_straightness <= 0.10:
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
    
    return frame_copy

    # cv2.imshow("Skeleton Image", frame_copy)
    # cv2.waitKey(0);


# In[51]:


# Reading the video from filepath provided above and passing it through the age clasification method defined above.
# Source 1: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# Source 2: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

# Creating a VideoCapture object.
cap = cv2.VideoCapture(my_video)

# Checking if video can be accessed successfully.
if (cap.isOpened() == False): 
    print("Unable to read video!")

# Defining a new dataframe to store the skeleton point coordinates from each frame of the video.
frame_counter = 1

while(cap.isOpened()):
    
    # Grabbing each individual frame, frame-by-frame.
    ret, frame = cap.read()
    
    if ret==True:
        
        if frame_counter == jump_squat_frame:
            print(f"{jump_squat_frame}")
            # jump_quality_at_squat_frame(frame, df.loc[frame_counter], df_imputed.loc[frame_counter], df_imputed_norm.loc[frame_counter])
            
            # new_my_frame = new_frame_name(my_video, frame_counter)
            # cv2.imwrite(new_my_frame, skeleton_frame)
            # print(f"Saved to {new_my_frame}")
            
        elif frame_counter == jump_peak_frame:
            skeleton_frame = jump_quality_at_peak_frame(frame, df.loc[frame_counter], df_imputed.loc[frame_counter], df_imputed_norm.loc[frame_counter])
            
            new_my_frame = new_frame_name(my_video, frame_counter)
            cv2.imwrite(new_my_frame, skeleton_frame)
            print(f"Saved to {new_my_frame}")
            
        elif frame_counter == jump_land_frame:
            print(f"{jump_land_frame}")
            # jump_quality_at_land_frame(frame, df.loc[frame_counter], df_imputed.loc[frame_counter], df_imputed_norm.loc[frame_counter])
            
            # new_my_frame = new_frame_name(my_video, frame_counter)
            # cv2.imwrite(new_my_frame, skeleton_frame)
            # print(f"Saved to {new_my_frame}")

        # Displaying the frame with age detected.
        # cv2.imshow("Output Video", skeleton_frame)
        
        frame_counter += 1
        
        # Exiting if "Q" key is pressed on the keyboard.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Releasing the VideoCapture object, and closing the displayed frame.
cap.release()
cv2.destroyAllWindows()
print("DONE!")