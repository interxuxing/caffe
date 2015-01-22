vgg_zoo_dir = '/media/DataDisk/myproject/deeplearning/caffe-zoo/VGG';
model_def_file = fullfile(vgg_zoo_dir, 'VGG_CNN_S_deploy.prototxt');
model_file = fullfile(vgg_zoo_dir, 'VGG_CNN_S.caffemodel');
mean_file = fullfile(vgg_zoo_dir, 'VGG_mean.mat');
use_gpu = false;


im = imread('../../examples/images/cat.jpg');

scores = matcaffe_demo_vgg(im, use_gpu, model_def_file, model_file, mean_file);

fprintf('finished! \n');