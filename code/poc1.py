import numpy as np
import pandas as pd
import cv2
import os
import sys
import getopt

import matplotlib.pyplot as plt
import seaborn as sns

def count_frames(my_video):
    # Creating a VideoCapture object.
    cap = cv2.VideoCapture(my_video)
    # Checking if video can be accessed successfully.
    if (cap.isOpened() == False):
        print("Unable to read video!")
        sys.exit(2)
    
    total_frames = 0
    while(cap.isOpened()):
        # Grabbing each individual frame, frame-by-frame.
        ret, frame = cap.read()
        if ret==True:
            total_frames += 1

        # Exiting if "Q" key is pressed on the keyboard.
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        else:
            break

    # Releasing the VideoCapture object.
    cap.release()

    return (total_frames)

def main(argv):

        try:
            opts, args = getopt.getopt(argv, "", ["inputfile=", "orient="]) 
        except getopt.GetoptError:
            print ('Usage: python poc1.py --inputfile=<file_path> --orient=<left>')
            sys.exit(2)

        opt1 = [i[0] for i in opts]

        if not all(x in opt1 for x in ['--inputfile', '--orient']):
            print ('Usage: python poc1.py --inputfile=<file_path> --orient=<left>')
            sys.exit(2)

        for opt, arg in opts:
            if opt == '--inputfile':
                my_video = arg
            if opt == '--orient':
                orient = arg
        
        #print ("Input file is: ", input_file)
        #print ("Orientation is: ", orient)
        
        #Count the total frames
        total_frames = count_frames(my_video)
        print ('Total {} frames'.format(str(total_frames)))

        
        #Report the status of the tables
        
        #Insert the data

        
        #Report DB Status
        
        
	
if __name__ == "__main__":
   main(sys.argv[1:])
