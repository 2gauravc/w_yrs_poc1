
from functions import general_functions as gf
from functions import db_functions as db
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score
from datetime import datetime
import os


#df = db.execute_sql_on_db()
#df.to_csv('./tmp/model_result.csv', index=False)

df = pd.read_csv('./tmp/model_result.csv')

temp_dir = './tmp'
filename1 = 'model_eval' + datetime.now().strftime("%Y%m%d-%H%M") +'.txt'
print_file_name = os.path.join(temp_dir, filename1)
print_file = open(print_file_name, "w")

# Confusion matrix
print ("Confusion Matrix", file=print_file)
print("--------------------------------------------------", file=print_file)
conf_matrix = df.pivot_table(index='detected_pose', columns='actual_pose', values='videofilename', aggfunc='count', margins=True)
print (conf_matrix, file=print_file)
print ("\n\n", file=print_file)

# Classification Report
print ("Classification Report", file=print_file)
print("--------------------------------------------------" , file=print_file)
class_report = classification_report(df.actual_pose, df.detected_pose, digits=2, output_dict = False)
print (class_report , file=print_file)
print ("\n" , file=print_file)


df_70 = df.loc[df.detected_pose_conf > 0.7]

# Classification Report for confidence > 70%
print ("Classification Report for confidence > 70%", file=print_file)
print("--------------------------------------------------" , file=print_file)
class_report = classification_report(df_70.actual_pose, df_70.detected_pose, digits=2, output_dict = False)
print (class_report , file=print_file)
print ("\n" , file=print_file)

df_90 = df.loc[df.detected_pose_conf > 0.9]

# Classification Report for confidence > 90%
print ("Classification Report for confidence > 90%", file=print_file)
print("--------------------------------------------------" , file=print_file)
class_report = classification_report(df_90.actual_pose, df_90.detected_pose, digits=2, output_dict = False)
print (class_report , file=print_file)
print ("\n" , file=print_file)


# Close the file
print_file.close()

# put the model evaluation file on S3

#if gf.upload_file_to_s3(print_file_name, 'w-yrs-model-metrics', filename1):
#    print ("All done.")
#else:
#    print ("Could not upload analyzed video to S3. Saved analyzed video to: {}".format(my_video.analysed_video_path))


