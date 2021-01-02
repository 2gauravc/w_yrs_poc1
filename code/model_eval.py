
#-------------------------------------------------------------
#Model Evaluation Report.
#This code evaluates 2 models 1. Critical Pose 2. Human Pose Points
# The Model evaluation files are written to ./tmp (2 separate files)
#-------------------------------------------------------


from functions import general_functions as gf
from functions import db_functions as db
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score
from datetime import datetime
import os


def conv_db_records_to_df_critical_pose(records):
    videofileName = []
    frame_no = []
    model_name = []
    model_version =  []
    detected_pose = []
    detected_pose_conf = []
    actual_pose = []
    
    for record in records:
        videofileName.append(record[0])
        frame_no.append(record[1])
        model_name.append(record[2])
        model_version.append(record[3])
        detected_pose.append(record[4])
        detected_pose_conf.append(record[5])
        actual_pose.append(record[6])
        
        df = pd.DataFrame ({'videofilename':videofileName,'frame_no':frame_no, 'model_name':model_name, 'model_version':model_version, 'detected_pose':detected_pose,'detected_pose_conf':detected_pose_conf, 'actual_pose':actual_pose})
    
    print("Found {} records in database".format(df.shape[0]))
    return df

def conv_db_records_to_df_human_pose(records):
    video_file_name = []
    frame_num = []
    model_name_s = []
    model_version_s = []
    confidence = []
    pt0_forehead_x = []
    pt0_forehead_y = []
    pt1_upperchest_x = []
    pt1_upperchest_y = []
    pt2_rightshoulder_x = []
    pt2_rightshoulder_y = []
    pt3_rightelbow_x = []
    pt3_rightelbow_y = []
    pt4_rightwrist_x = []
    pt4_rightwrist_y = []
    pt5_leftshoulder_x = []
    pt5_leftshoulder_y = []
    pt6_leftelbow_x = []
    pt6_leftelbow_y = []
    pt7_leftwrist_x = []
    pt7_leftwrist_y = []
    pt8_navel_x = []
    pt8_navel_y = []
    pt9_righthip_x = []
    pt9_righthip_y = []
    pt10_rightknee_x = []
    pt10_rightknee_y = []
    pt11_rightankle_x = []
    pt11_rightankle_y= []
    pt12_lefthip_x = []
    pt12_lefthip_y = []
    pt13_leftknee_x = []
    pt13_leftknee_y = []
    pt14_leftankle_x = []
    pt14_leftankle_y = []
    
    for record in records:
        video_file_name.append(record[0])
        frame_num.append(record[1])
        model_name_s.append(record[2])
        model_version_s.append(record[3])
        confidence.append(record[4])
        pt0_forehead_x.append(record[5])
        pt0_forehead_y.append(record[6])
        pt1_upperchest_x.append(record[7])
        pt1_upperchest_y.append(record[8])
        pt2_rightshoulder_x.append(record[9])
        pt2_rightshoulder_y.append(record[10])
        pt3_rightelbow_x.append(record[11])
        pt3_rightelbow_y.append(record[12])
        pt4_rightwrist_x.append(record[13])
        pt4_rightwrist_y.append(record[14])
        pt5_leftshoulder_x.append(record[15])
        pt5_leftshoulder_y.append(record[16])
        pt6_leftelbow_x.append(record[17])
        pt6_leftelbow_y.append(record[18])
        pt7_leftwrist_x.append(record[19])
        pt7_leftwrist_y.append(record[20])
        pt8_navel_x.append(record[21])
        pt8_navel_y.append(record[22])
        pt9_righthip_x.append(record[23])
        pt9_righthip_y.append(record[24])
        pt10_rightknee_x.append(record[25])
        pt10_rightknee_y.append(record[26])
        pt11_rightankle_x.append(record[27])
        pt11_rightankle_y.append(record[28])
        pt12_lefthip_x.append(record[29])
        pt12_lefthip_y.append(record[30])
        pt13_leftknee_x.append(record[31])
        pt13_leftknee_y.append(record[32])
        pt14_leftankle_x.append(record[33])
        pt14_leftankle_y.append(record[34])

    
    df_pose = pd.DataFrame({'video_file_name':video_file_name, 'frame_num':frame_num, 'model_name':model_name_s,'model_version':model_version_s, 'confidence':confidence, 'pt0_forehead_x':pt0_forehead_x,'pt0_forehead_y':pt0_forehead_y, 'pt1_upperchest_x':pt1_upperchest_x, 'pt1_upperchest_y':pt1_upperchest_y, 'pt2_rightshoulder_x':pt2_rightshoulder_x, 'pt2_rightshoulder_y':pt2_rightshoulder_y, 'pt3_rightelbow_x':pt3_rightelbow_x, 'pt3_rightelbow_y':pt3_rightelbow_y,'pt4_rightwrist_x':pt4_rightwrist_x, 'pt4_rightwrist_y':pt4_rightwrist_y, 'pt5_leftshoulder_x':pt5_leftshoulder_x, 'pt5_leftshoulder_y':pt5_leftshoulder_y, 'pt6_leftelbow_x':pt6_leftelbow_x,'pt6_leftelbow_y':pt6_leftelbow_y, 'pt7_leftwrist_x':pt7_leftwrist_x, 'pt7_leftwrist_y':pt7_leftwrist_y, 'pt8_navel_x':pt8_navel_x, 'pt8_navel_y':pt8_navel_y, 'pt9_righthip_x':pt9_righthip_x, 'pt9_righthip_y':pt9_righthip_y, 'pt10_rightknee_x':pt10_rightknee_x, 'pt10_rightknee_y':pt10_rightknee_y, 'pt11_rightankle_x':pt11_rightankle_x, 'pt11_rightankle_y':pt11_rightankle_y, 'pt12_lefthip_x':pt12_lefthip_x, 'pt12_lefthip_y':pt12_lefthip_y, 'pt13_leftknee_x':pt13_leftknee_x, 'pt13_leftknee_y':pt13_leftknee_y, 'pt14_leftankle_x':pt14_leftankle_x, 'pt14_leftankle_y':pt14_leftankle_y})
        
        
        
    print("Found {} records in human pose details table".format(df_pose.shape[0]))
    return df_pose

def evaluate_critical_pose_model(print_file_name):
    print_file = open(print_file_name, "w")
    # 1. Evaluate the critical pose detection model
    #----------------------------------------------------------------------

    print ("1. Evalute critical pose model", file=print_file)
    print("--------------------------------------------------", file=print_file)
    print("--------------------------------------------------", file=print_file)
    
    
    #sql_text = "select * from video_frame_vjump_pose;"
    #recs = db.execute_sql_on_db(sql_text)
    #df = conv_db_records_to_df_critical_pose(recs)
    #df.to_csv('./tmp/critical_pose_model_result.csv', index=False)
    
    df_pose = pd.read_csv('./tmp/critical_pose_model_result.csv')
    

    
    # Confusion matrix
    print ("Confusion Matrix", file=print_file)
    print("--------------------------------------------------", file=print_file)
    conf_matrix = df_pose.pivot_table(index='detected_pose', columns='actual_pose', values='videofilename', aggfunc='count', margins=True)
    print (conf_matrix, file=print_file)
    print ("\n\n", file=print_file)
    
    # Classification Report
    print ("Classification Report", file=print_file)
    print("--------------------------------------------------" , file=print_file)
    class_report = classification_report(df_pose.actual_pose, df_pose.detected_pose, digits=2, output_dict = False)
    print (class_report , file=print_file)
    print ("\n" , file=print_file)
    
    
    df_70 = df_pose.loc[df_pose.detected_pose_conf > 0.7]
    
    # Classification Report for confidence > 70%
    print ("Classification Report for confidence > 70%", file=print_file)
    print("--------------------------------------------------" , file=print_file)
    class_report = classification_report(df_70.actual_pose, df_70.detected_pose, digits=2, output_dict = False)
    print (class_report , file=print_file)
    print ("\n" , file=print_file)
    
    df_90 = df_pose.loc[df_pose.detected_pose_conf > 0.9]
    
    # Classification Report for confidence > 90%
    print ("Classification Report for confidence > 90%", file=print_file)
    print("--------------------------------------------------" , file=print_file)
    class_report = classification_report(df_90.actual_pose, df_90.detected_pose, digits=2, output_dict = False)
    print (class_report , file=print_file)
    print ("\n" , file=print_file)
    
    
    # Close the file
    print_file.close()
    
    return(df_pose)

def f(x):
    # total_frames
    # squat_frames_actual
    # squat_frames_detected
    # squat_frames_detected_with_cpp_available (cpp : Critical Pose Points)
    # squat_frames_detected_with_cpp_available_actual_squat
    # jump_peak_frames_actual
    # jump_peak_frames_detected
    # jump_peak_frames_detected_with_cpp_available
    # jump_peak_frames_detected_with_cpp_available_actual_jump_peak
    # landing_frames_actual
    # landing_frames_detected
    # landing_frames_detected_with_cpp_available
    # landing_frames_detected_with_cpp_available_actual_landing
    
    d = {}
    d['total_frames'] = x['frame_num'].nunique()
    d['squat_frames'] = x.loc[(x['actual_pose'] == 'squat'), 'frame_num'].count()
    d['squat_frames_detected'] = x.loc[( (x['detected_pose'] == 'squat') & (x['actual_pose'] == 'squat') ), 'frame_num'].count()
    d['squat_frames_detected_cpp_l'] = x.loc[( (x['detected_pose'] == 'squat') & (x['actual_pose'] == 'squat') & (x['all_cr_pp_avail_left']) ), 'frame_num'].count()
    d['squat_frames_detected_cpp_r'] = x.loc[( (x['detected_pose'] == 'squat') & (x['actual_pose'] == 'squat') & (x['all_cr_pp_avail_right']) ), 'frame_num'].count()
    
    d['jump_peak_frames'] = x.loc[(x['actual_pose'] == 'jump_peak'), 'frame_num'].count()
    d['jump_peak_frames_detected'] = x.loc[( (x['detected_pose'] == 'jump_peak') & (x['actual_pose'] == 'jump_peak') ), 'frame_num'].count()
    d['jump_peak_frames_detected_cpp_l'] = x.loc[( (x['detected_pose'] == 'jump_peak') & (x['actual_pose'] == 'jump_peak') & (x['all_cr_pp_avail_left']) ), 'frame_num'].count()
    d['jump_peak_frames_detected_cpp_r'] = x.loc[( (x['detected_pose'] == 'jump_peak') & (x['actual_pose'] == 'jump_peak') & (x['all_cr_pp_avail_right']) ), 'frame_num'].count()

    d['landing_frames'] = x.loc[(x['actual_pose'] == 'landing'), 'frame_num'].count()
    d['landing_frames_detected'] = x.loc[( (x['detected_pose'] == 'landing') & (x['actual_pose'] == 'landing') ), 'frame_num'].count()
    d['landing_frames_detected_cpp_l'] = x.loc[( (x['detected_pose'] == 'landing') & (x['actual_pose'] == 'landing') & (x['all_cr_pp_avail_left']) ), 'frame_num'].count()
    d['landing_frames_detected_cpp_r'] = x.loc[( (x['detected_pose'] == 'landing') & (x['actual_pose'] == 'landing') & (x['all_cr_pp_avail_right']) ), 'frame_num'].count()
    
    return pd.Series(d, index=['total_frames', 'squat_frames','squat_frames_detected', 'squat_frames_detected_cpp_l', 'squat_frames_detected_cpp_r', 'jump_peak_frames', 'jump_peak_frames_detected', 'jump_peak_frames_detected_cpp_l', 'jump_peak_frames_detected_cpp_r', 'landing_frames', 'landing_frames_detected', 'landing_frames_detected_cpp_l', 'landing_frames_detected_cpp_r' ])



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

temp_dir = './tmp'
filename1 = 'model_eval' + datetime.now().strftime("%Y%m%d-%H%M") +'.txt'
print_file_name = os.path.join(temp_dir, filename1)

df_cr_pose = evaluate_critical_pose_model(print_file_name)
print("Critical Pose Model has {} rows".format(df_cr_pose.shape[0]))
#print(df_cr_pose.head())

# put the model evaluation file on S3

#if gf.upload_file_to_s3(print_file_name, 'w-yrs-model-metrics', filename1):
#    print ("All done.")
#else:
#    print ("Could not upload analyzed video to S3. Saved analyzed video to: {}".format(my_video.analysed_video_path))



# 2. Evaluate the Human pose point model
#----------------------------------------------------------------------

# 1. Add the following attributes to each row (frame) of the 'video_pose_proc_details' table:
# num_pose_points_available
# all_critical_pose_points_available_left (0/1) (shoulder, elbow, wrist, hip, knee, ankle)
# all_critical_pose_points_available_right (0/1) (shoulder, elbow, wrist, hip, knee, ankle)

#2. Join the above dataframe with the video_frame_vjump_pose dataframe

#3. Summarize for each video:

# total_frames
# squat_frames_actual
# squat_frames_detected
# squat_frames_detected_with_cpp_available (cpp : Critical Pose Points)
# squat_frames_detected_with_cpp_available_actual_squat
# jump_peak_frames_actual
# jump_peak_frames_detected
# jump_peak_frames_detected_with_cpp_available
# jump_peak_frames_detected_with_cpp_available_actual_jump_peak
# landing_frames_actual
# landing_frames_detected
# landing_frames_detected_with_cpp_available
# landing_frames_detected_with_cpp_available_actual_landing


#sql_text = "select * from video_pose_proc_details;"
#recs = db.execute_sql_on_db(sql_text)
#print("Found {} records in DB".format(len(recs)))
#df_pose = conv_db_records_to_df_human_pose(recs)
#df_pose.to_csv('./tmp/human_pose_model_result.csv', index=False)

df_hu_pose = pd.read_csv('./tmp/human_pose_model_result.csv')

#temp_dir = './tmp'
#print_file_name = os.path.join(temp_dir, filename1)
#print_file = open(print_file_name, "a")


# 1. Add attributes to each row (frame) of the 'video_pose_proc_details' table

all_pose_points = ['pt0_forehead_x','pt0_forehead_y', 'pt1_upperchest_x', 'pt1_upperchest_y', 'pt2_rightshoulder_x', 'pt2_rightshoulder_y', 'pt3_rightelbow_x', 'pt3_rightelbow_y', 'pt4_rightwrist_x','pt4_rightwrist_y', 'pt5_leftshoulder_x', 'pt5_leftshoulder_y', 'pt6_leftelbow_x', 'pt6_leftelbow_y', 'pt7_leftwrist_x', 'pt7_leftwrist_y', 'pt8_navel_x', 'pt8_navel_y', 'pt9_righthip_x', 'pt9_righthip_y', 'pt10_rightknee_x', 'pt10_rightknee_y', 'pt11_rightankle_x', 'pt11_rightankle_y', 'pt12_lefthip_x', 'pt12_lefthip_y', 'pt13_leftknee_x', 'pt13_leftknee_y', 'pt14_leftankle_x', 'pt14_leftankle_y']

critical_pose_points_left = ['pt5_leftshoulder_x','pt6_leftelbow_x', 'pt7_leftwrist_x','pt12_lefthip_x','pt13_leftknee_x','pt14_leftankle_x']

critical_pose_points_right = ['pt2_rightshoulder_x','pt3_rightelbow_x', 'pt4_rightwrist_x','pt9_righthip_x','pt10_rightknee_x','pt11_rightankle_x']


df_trans = df_hu_pose[all_pose_points].applymap(lambda x: 0 if x==-1 else 1)
df_hu_pose['num_pp_avail'] = df_trans.sum(axis = 1)
df_hu_pose['all_cr_pp_avail_left'] = df_trans[critical_pose_points_left].apply(lambda x: 0 if any(x==0) else 1, axis =1)
df_hu_pose['all_cr_pp_avail_right'] = df_trans[critical_pose_points_right].apply(lambda x: 0 if any(x==0) else 1, axis =1)

#df_pose.to_csv('./tmp/df_pose.csv', index=0)

print("Human Pose Model has Rows", df_hu_pose.shape[0])


#2. Join the above dataframe with the video_frame_vjump_pose dataframe

#cols_to join_hp = ['video_file_name','frame_num','model_name','model_version','confidence','num_pp_avail','all_cr_pp_avail_left', 'all_cr_pp_avail_right']

df_merged = pd.merge(df_hu_pose, df_cr_pose, how = 'left' ,left_on = ['video_file_name', 'frame_num'], right_on = ['videofilename', 'frame_no'])

#df_merged.to_csv('./tmp/df_merged.csv', index=False)

print ("Merge completed. Merged DF has {} rows".format(df_merged.shape[0]))


#Step 3. Summarize stats for each video:

cols_to_keep = ['video_file_name', 'frame_num', 'confidence', 'actual_pose', 'detected_pose', 'detected_pose_conf', 'num_pp_avail', 'all_cr_pp_avail_left', 'all_cr_pp_avail_right']
df_merged_a = df_merged[cols_to_keep]




df_vid_summ = df_merged_a.groupby('video_file_name').apply(f).reset_index()


#Calculate the key percentages

df_vid_summ['cp_acc_squat'] = round(df_vid_summ['squat_frames_detected'] / df_vid_summ['squat_frames'],2)
df_vid_summ['cp_acc_jump_peak'] = round (df_vid_summ['jump_peak_frames_detected'] / df_vid_summ['jump_peak_frames'],2)
df_vid_summ['cp_acc_landing'] = round (df_vid_summ['landing_frames_detected'] / df_vid_summ['landing_frames'],2)

df_vid_summ['hp_avail_squat'] = round(df_vid_summ[['squat_frames_detected_cpp_l', 'squat_frames_detected_cpp_r']].max(axis=1) / df_vid_summ['squat_frames_detected'] ,2)
df_vid_summ['hp_avail_jump_peak'] = round(df_vid_summ[['jump_peak_frames_detected_cpp_l','jump_peak_frames_detected_cpp_r']].max(axis=1) / df_vid_summ['jump_peak_frames_detected'] ,2)
df_vid_summ['hp_avail_landing'] = round(df_vid_summ[['landing_frames_detected_cpp_l','landing_frames_detected_cpp_r']].max(axis=1) / df_vid_summ['landing_frames_detected'] ,2)

cols_to_print = ['video_file_name', 'cp_acc_squat', 'cp_acc_jump_peak', 'cp_acc_landing', 'hp_avail_squat', 'hp_avail_jump_peak', 'hp_avail_landing']
df_vid_summ_print = df_vid_summ[cols_to_print]
    

print_file = open(print_file_name, "a")

print ("2. Evaluate human pose point  model", file=print_file)
print("--------------------------------------------------", file=print_file)
print("--------------------------------------------------", file=print_file)


df_avail_pp = df_trans.sum(axis = 0)

print(df_avail_pp/df_trans.shape[0], file=print_file)


print ("3. Evaluate report feasibility", file=print_file)
print("--------------------------------------------------", file=print_file)
print("--------------------------------------------------", file=print_file)

print ("Pose Point Availability", file=print_file)
print("--------------------------------------------------", file=print_file)

print (df_vid_summ_print, file=print_file)
print ("\n\n", file=print_file)

# Close the file
print_file.close()