"""
    exampleFeatureExtractionCeffe.py is a demo script to extract Caffe deep nueral network features
    __author__ = 'limu'
"""


'''
This script extracts features of images using Caffe's AlexNet and outputs a numpy array

Usage:
    python caffe_features.py /input/folder/ /output/file [-d GPU|CPU] [-f jpg|png|...]

By:
    @edersantana

License:
    [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE)
'''

import os
import leveldb
import caffe
from caffe.proto import caffe_pb2
import numpy as np
from glob import glob
from natsort import natsorted
import argparse

def extract(input_path, device='GPU', format='jpg'):

    features = np.zeros((0,4096))

    # get the abstract path of current dir
    abs_current_dir = os.path.abspath(os.path.curdir)
    # Prepare a list of images
    # get abstract path of input_path
    abs_path = os.path.abspath(input_path)
    files = natsorted(glob(os.path.join(abs_path, '*.'+format)))

    '''
    Here is the hack. Caffe feature extraction seems to work with 500 images at a time.
    Thus, we dump only 500 images at a time to file_list.txt
    '''

    # Create temp folder for current features
    os.system(('rm -rf %s/_temp' % abs_current_dir))
    os.system(('mkdir %s/_temp' % abs_current_dir))

    for idx in range(0, 1+len(files)/500):



        first = idx * 500
        last  = min((idx+1) * 500, len(files))
        file_batch = files[first:last]

        file_list = open(('%s/_temp/file_list.txt' % abs_current_dir),'w') # delete the previous list, please
        for l in file_batch:
            file_list.write(l + ' 0\n')

        file_list.close()

        # Extract features
        caffe_root = os.environ['CAFFE_ROOT']
        p1 = os.path.join(caffe_root, 'build/tools/extract_features.bin')
        p2 = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

        # move the prototxt file to the temp root dirctory
        p3 = os.path.join(abs_current_dir, 'imagenet_val.prototxt')
        os.system('cp %s %s/_temp' % (p3, abs_current_dir))
        #p3 = ' ./imagenet_val.prototxt fc7 ../_temp/features%d 10 %s' % (idx, device)
        p4 = ' %s/_temp/imagenet_val.prototxt fc7 %s/_temp/features%d 10 %s' % (abs_current_dir, abs_current_dir, idx, device)
        cmd3 = p1 + ' ' + p2 + p4
        print cmd3
        os.system(cmd3)

        """
        # Load features to numpy array
        print '\n\n LOADING FEATURES TO NUMPY ARRAY. BATCH # : %d/%d' % (idx+1, 1+len(files)/500)
        db = leveldb.LevelDB('%s/_temp/features%d/' % (abs_current_dir, idx))

        for k in range(len(file_batch)):
            datum = caffe_pb2.Datum.FromString(db.Get(str(k)))
            features = np.vstack([features, caffe.io.datum_to_array(datum)[:,:,0]])

        del db

        # delete the current features-idx directory
        print 'remove current feature%d directory' % idx
        os.system(('rm -rf %s/_temp/features%d' % (abs_current_dir, idx)))
        """

    # Delete temp folder
    #print 'remove temp folder %s/_temp' % abs_current_dir
    #os.system(('rm -rf %s/_temp' % abs_current_dir))

    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', metavar='input_path', type=str, help='/path/to/input/jpg/images/')
    parser.add_argument('output_path', metavar='output_path', type=str, help='/path/to/save/numpy/array/at')
    parser.add_argument('-d', metavar='device', type=str, default='GPU', help='[GPU* | CPU]')
    parser.add_argument('-f', metavar='format', type=str, default='jpg', help='image format')
    args = parser.parse_args()

    F = extract(args.input_path, args.d, args.f)


    # relative path is OK
    print 'Saving numpy array as %s.npy' % args.output_path
    #np.save(args.output_path, F)