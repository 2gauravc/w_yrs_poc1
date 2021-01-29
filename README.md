# w_yrs_poc1
This is my first readme


# AWS Setup

Virtual Machine Hardware Specifications
1 Ubuntu 18.04 
2 HDD 20 GB 

Software Specifications

	
# Install Guide

Once in the AWS virtual machine, do the following steps
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh


Create and activate python env

Create and Activate Env
	---------------------------------------
	    6  conda create -n fastai-dev python=3.6
	    7  conda activate fastai-dev
 

Install packages:
cv2, scipy, boto3,shutil, pandas

pip install <<package_name>

B) Git clone the w_yrs_poc1 repo 

B). AWS Configuration

GC to create IAM Role for AJ. AJ to use the credentials to configure aws (using aws CLI)

C). How to Run the code: 

On a ubuntu (linux) machine inside code/. 
python3 pipeline_critical_pose.py --file=<<ilename_in_s3_bkt_w-yrs-input-video>> 
--rotate <<angle>>


# How to Build Pipelines



