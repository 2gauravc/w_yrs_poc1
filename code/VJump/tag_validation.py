import time
import logging
import argparse
import os 
from os import path
import sys
import pandas as pd

from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

def main(argv):

	# Find the target folder 
	ap = argparse.ArgumentParser()
	ap.add_argument("--folder", required=True, help="folder with tagged files")
	args = vars(ap.parse_args())	
	folder = args["folder"]
	print ("Picking up files from  {} folder".format(folder))
	if path.isdir(folder) == False:
		print("Folder path not found. Exiting...")
		sys.exit(1)
	
	filenames = find_csv_filenames(folder)

	for filename in filenames:
		print ("Start processing for file {}".format(filename))
		
		# Read the file 
		df = pd.read_csv(os.path.join(folder, filename))
		df = df.set_index('frames')
		
		# Report number of frames found
		num_frames = df.shape[0]
		print ("\tFound {} frames".format(num_frames))

		#Replace NA with zeroes 
		df = df.fillna(0)
		err_frames = 0
		seq = []
		seq_frames = []
		cur_seq_frames = 0
		frameserr_text = ""

		for index, frame in df.iterrows():
			# Process one frame at a time 
			frameerr_text = ""
			errs_in_frame = 0
			framepos = frame
			
			#All values should be either 0 or 1 
			if framepos.isin([0,1]).sum() != 3: 
				errs_in_frame+=1
				frameerr_text += "\tError in frame {}: Not all values are 0 or 1.".format(index)
				
			#Only 1 position can be 1
			if len(framepos.iloc[framepos.to_numpy().nonzero()[0]]) > 1:
				errs_in_frame +=1
				frameerr_text += "\n\tError in frame {}: More than 1 position has non zero value.\n".format(index)

			if errs_in_frame > 0: 
				err_frames +=1 	
			
			# Create the sequence frame and count frame
			if framepos['squat'] == 1:
				cur_frame_pos = 's'
			elif framepos['jump_peak'] == 1:
				cur_frame_pos = 'j'
			elif framepos['landing'] == 1: 
				cur_frame_pos = 'l'
			else:
				cur_frame_pos = 'N'

			# Find out if the sequence is continuing or changing
			if not seq: #Sequence is blank and we are starting off
				if cur_frame_pos != '': 
					seq.append(cur_frame_pos)
					cur_seq_frames += 1
			else:
				prev_frame_pos = seq[-1] #previous value 
				if prev_frame_pos == cur_frame_pos: 
					cur_seq_frames +=1
					if index == num_frames: # this is the last frame 
						seq_frames.append(cur_seq_frames)	
				else: #sequence is changing 
					seq.append(cur_frame_pos)
					seq_frames.append(cur_seq_frames)
					cur_seq_frames = 1
					if index == num_frames: # this is the last frame 
						seq_frames.append(cur_seq_frames)	
			
			frameserr_text += frameerr_text
				
		# Create a dataframe of pose sequence and length (frames)
		df_dict = {'pose':seq, 'num_frames':seq_frames}
		df = pd.DataFrame(df_dict) 

		#Find min,max, avg number of frames per pose sequence 
		df_agg = df.groupby('pose').agg(num_seqs=('num_frames','count'), avg_frames_per_seq = ('num_frames','mean'),
		min_frames_per_seq = ('num_frames','min'), max_frames_per_seq = ('num_frames','max'))

		# Find errors or warnings in the sequences 
	
		# Print the number of errors and error text for all frames 

		print ("\tFound {} frames with errors".format(err_frames))
		print("\t{}".format(frameserr_text))


		

	

if __name__ == "__main__":
	main(sys.argv[1:])