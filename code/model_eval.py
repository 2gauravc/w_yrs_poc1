
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
print_file = os.path.join(temp_dir, filename1)
print_file = open(print_file, "w")

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




