#!/bin/bash
# This is a simple bash script to run the demo_vgg.m matlab script with input paramters
#	and predict the top-5 label indexes with vgg models.
# This script is call in the python code of app_limu_vgg.py

# usage of this script
# ./vgg_classify.sh model_def_file model_file mean_file img_file
# here the paths are full path
#
#
#


# model_def_file=/media/DataDisk/myproject/deeplearning/caffe/models/VGG/VGG_CNN_S_deploy.prototxt
# model_file=/media/DataDisk/myproject/deeplearning/caffe/models/VGG/VGG_CNN_S.caffemodel
# mean_file=/media/DataDisk/myproject/deeplearning/caffe/models/VGG/VGG_mean.mat
# img_file=/media/DataDisk/myproject/deeplearning/caffe/examples/images/cat.jpg

model_def_file=${1}
model_file=${2}
mean_file=${3}
img_file=${4}

# initialize for some variables and path
matlab_exec=matlab
vgg_script=vgg_classify
vgg_script_dir=/media/DataDisk/myproject/deeplearning/caffe/matlab/caffe

cd ${vgg_script_dir}
chmod +x ${vgg_script}.m

matlab -nojvm -nodisplay -nosplash -r \
"${vgg_script}('${model_def_file}', '${model_file}', '${mean_file}', '${img_file}')"