function vgg_classify(model_def_file, model_file, mean_file, img_file)

if nargin < 1
% use basic model from zoo   
    vgg_zoo_dir = '/media/DataDisk/myproject/deeplearning/caffe-zoo/VGG';
    model_def_file = fullfile(vgg_zoo_dir, 'VGG_CNN_S_deploy.prototxt');
    model_file = fullfile(vgg_zoo_dir, 'VGG_CNN_S.caffemodel');
    mean_file = fullfile(vgg_zoo_dir, 'VGG_mean.mat');
    img_file = '/media/DataDisk/myproject/deeplearning/caffe/examples/images/cat.jpg';
elseif nargin == 4
%     here are full path
%     model_def_file = fullfile(vgg_zoo_dir, model_def_file);
%     model_file = fullfile(vgg_zoo_dir, model_file);
%     mean_file = fullfile(vgg_zoo_dir, mean_file); 
else
    error('wrong input paramters, please check again!')
end


use_gpu = true;

im = imread(img_file);

scores = matcaffe_demo_vgg(im, use_gpu, model_def_file, model_file, mean_file);

% now sort to return the top-5 labels
scores_mean = mean(scores,2);
 
% [scores_idx,labels_idx] = sort(scores_mean, 1, 'descend');
% top5_labels = labels_idx(1:5);
% top5_scores = scores_idx(1:5);
% 
% if ~isempty(top5_labels)
%     dlmwrite('predLabelsIdx.txt', [top5_labels, top5_scores], ' ');
% end

% only write the scores to file
if ~isempty(scores)
    dlmwrite('predLabelsIdx.txt', scores_mean); % without sort
end

fprintf('predict labels for image %s finished! \n', img_file);

quit;