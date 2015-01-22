import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import cStringIO as StringIO
import urllib
import caffe
import exifutil
import csv


def read_labels_csv(file_csv, file_imagenetsynsets):
    '''
        read_labels_csv is a function to read (idx, synsetid/label) from hybridcnnplace csv style file
    :param file_csv: filename of the csv file
    :param file_imagenetsynsets: filename of the imagenet sysets file
    :return: pairs of (idx, synsetid/label)
    '''
    with open(file_imagenetsynsets) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
    labels = labels_df.sort('synset_id')['name'].values

#    # read csv file
    Num_synsets = 978
    Num_places = 205

    labels_csv = np.zeros([Num_synsets + Num_places], dtype='|S8')

    # read csv file, and map the
    f = open(file_csv, 'rb')
    # first read synsets
    for i in range(Num_synsets):
        content = f.readline().strip()
        synset_id_i = content.split(' ')[0]

        # find the name of synset_id_i in labels_df
        res = (labels_df.get('synset_id') == synset_id_i)
        name = labels_df.get('name')[res]

        if name.empty:
            print 'no synset id %s found !' % synset_id_i

        labels_csv[i] = name[name.index[0]]

    # then read the places
    for j in range(Num_places):
        content = f.readline().strip().split(' ')[0]
        # further split with '/'
        place = content.split('/')[1]
        labels_csv[j+Num_synsets] = place
    # return the ndarray of strings of lables in csv
    return labels_csv

# program starts from here
if __name__ == '__main__':
    file_csv = '../../models/places_cnn_mit/categoryIndex_hybridCNN.csv'
    file_imagenetsynsets = '../../data/ilsvrc12/synset_words.txt'

    read_labels_csv(file_csv, file_imagenetsynsets)
