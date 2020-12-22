
###################################################################
# This file has a list of generic functions that are reusable across
# different pipelines.

###################################################################


import cv2
import sys
import argparse
import os
from scipy import ndimage
import boto3
import shutil


def check_video_file_open(video_path):
    video_read_successfully = False

    try:
        cap = cv2.VideoCapture(video_path)
        if (cap.isOpened() == True):
            video_read_successfully = True
        else:
            print ("Unable to read video")
    except:
        print ("Exception in trying to read video file")

    finally:
        cap.release()
        return (video_read_successfully)
    
def upload_file_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3')

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except ClientError as e:
        print("Error")
        return False

def download_file_from_s3(bucket,s3_file, local_file_path):
    print('Downloading file from S3...')
    s3 = boto3.resource('s3')

    try:
        s3.Bucket(bucket).download_file(s3_file, local_file_path)
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except ClientError as e:
        print("Error")
        return False

def compress_video_ffmpeg(old_video_path, new_video_path):
    k = os.system("ffmpeg -i {} -vcodec libx264 -crf 20 {}".format(old_video_path,new_video_path))
    return k
    
    

