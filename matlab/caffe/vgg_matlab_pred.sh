#!/bin/bash
# This is a simple bash script to run the demo_vgg.m matlab script with input paramters
#	and predict the top-5 label indexes with vgg models.

# usage of this script
# ./vgg_matlab_pred.sh vgg_zoo_dir model_def_file model_file mean_file img_file
# or ./vgg_matlab_pred.sh without any argments (which will use the default 
# 	models in demo_vgg.m file)

#
#
#


vgg_zoo_dir="/media/DataDisk/myproject/deeplearning/caffe-zoo/VGG"
model_def_file="VGG_CNN_S_deploy.prototxt"
model_file="VGG_CNN_S.caffemodel"
mean_file="VGG_mean.mat"
img_file="/media/DataDisk/myproject/deeplearning/caffe/examples/images/cat.jpg"

# initialize for some variables and path
matlab_exec=matlab
vgg_script=demo_vgg
vgg_script_dir=/media/DataDisk/myproject/deeplearning/caffe/matlab/caffe

cd ${vgg_script_dir}
chmod +x ${vgg_script}.m

matlab -nojvm -nodisplay -nosplash -r \
"${vgg_script}('${vgg_zoo_dir}', '${model_def_file}', '${model_file}', '${mean_file}', '${img_file}')"