# Running the Pipelines 

This repo has the following key Pipelines: 

- pipeline_critical_pose.py
- pipeline_human_pose_points.py
- pipeline-vjump-report.py

## pipeline_critical_pose.py

### EC2 Setup

Virtual Machine Hardware Specifications
1 Ubuntu 18.04 
2 HDD 20 GB 

### Software Install Guide

Once in the AWS virtual machine, do the following steps

#### Install Conda 
```
* wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
* chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
* rm ./Miniconda3-latest-Linux-x86_64.sh
```
#### Create and activate python environment
```
* conda create -n fastai-dev python=3.6
* conda activate fastai-dev
```

#### Install packages:

cv2, scipy, boto3,shutil, pandas
pip install <<package_name>

#### Git clone the w_yrs_poc1 repo 

### AWS Configuration

```
aws configure
```

### Running the Code 

``` 
python3 pipeline_critical_pose.py --file=<<filename_in_s3_bkt_w-yrs-input-video>> --rotate <<angle>> -o -d
```
