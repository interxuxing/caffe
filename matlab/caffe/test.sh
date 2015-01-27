#!/bin/bash

# a sample script that uses shell to call the matlab function with arguments.

filename=mat_prog_1
filedir=/media/DataDisk/myproject/deeplearning/caffe/matlab/caffe
cd ${filedir}
cat > ${filename}.m << EOF
function $filename(arg1, arg2)
% It is matlab script
fprintf('%s, %s \n', num2str(arg1), num2str(arg2))
disp('Hello World')
A=zeros(3,3);
if size(A)==3
disp('Hello again')
end
EOF
chmod +x ${filename}.m
#matlab -nodesktop -nosplash -nodisplay -r "run ./${filename}.m(2,3); quit;"
matlab -nojvm -nodisplay -nosplash -r "${filename}('20', 30)"