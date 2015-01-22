#!/bin/sh

#/media/DataDisk/myproject/deeplearning/caffe/build/tools/extract_features.bin
#/media/DataDisk/myproject/deeplearning/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
#/media/DataDisk/myproject/project-ieej2015/accv2014-code/bmrm_demo/python-script/_temp/batch_features/imagenet_val.prototxt fc7
#/media/DataDisk/myproject/project-ieej2015/accv2014-code/bmrm_demo/python-script/_temp/batch_features/features0 20 GPU


CAFFE_BINARY=/media/DataDisk/myproject/deeplearning/caffe/build/tools/extract_features.bin
CAFFE_MODEL=/media/DataDisk/myproject/deeplearning/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

RUN_ROOT=/media/DataDisk/myproject/project-ieej2015/accv2014-code/bmrm_demo/python-script/_temp

# now loop to run caffe feature extraction on bactch filelist
for rep in `seq 0 149`;
do
	echo "extract features for batch file ${rep}"
	rm ${RUN_ROOT}/batch_features/file_list.txt
	# copy batch file list to batch_features folder
	cp ${RUN_ROOT}/batchfiles/batchfilelist${rep}.txt ${RUN_ROOT}/batch_features/file_list.txt

	# run the script for current batch features
    ${CAFFE_BINARY} ${CAFFE_MODEL} ${RUN_ROOT}/batch_features/imagenet_val.prototxt fc7 ${RUN_ROOT}/batch_features/features${rep} 20 GPU
done
