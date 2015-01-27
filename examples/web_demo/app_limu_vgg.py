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
import subprocess
import skimage.io

REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

VGG_CLASSIFY_DIR = '/media/DataDisk/myproject/deeplearning/caffe/matlab/caffe'
# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index_limu.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

        # now write stringio to file
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + 'url_image.jpg'
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        # use skimage to save image array
        skimage.io.imsave(filename, image)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index_limu.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(filename) # here we use the filename, not image itself
    # print result
    return flask.render_template(
        'index_limu.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        print '1'
        imagefile = flask.request.files['imagefile']
        print '2'
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        print '3'
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        print 'Upload file name: ' + filename
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index_limu.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    print 'Begin to classify image %s' % filename
    result = app.clf.classify_image(filename) # here we use filename, not image
    return flask.render_template(
        'index_limu.html', has_result=True, result=result,
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = PILImage.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    # this is the old version path, model is saved in examples directory
    ''' 
    default_args = {
        'model_def_file': (
            '{}/examples/imagenet/imagenet_deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/examples/imagenet/caffe_reference_imagenet_model'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    '''
    # this is the new version path, model is saved in models/submodel/xxx directory
    '''
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    '''
    # here we use the vgg model for default args
    default_args = {
        'model_def_file': (
            '{}/models/VGG/VGG_ILSVRC_16_layers_deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/VGG/VGG_ILSVRC_16_layers.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/models/VGG/VGG_mean_16_layers.mat'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/models/VGG/caffe_ilsvrc12_vgg/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }

    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 227
    default_args['raw_scale'] = 255.
    default_args['gpu_mode'] = False

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        # logging.info('Loading net and associated files...')
        # self.net = caffe.Classifier(
        #     model_def_file, pretrained_model_file,
        #     image_dims=(image_dim, image_dim), raw_scale=raw_scale,
        #     mean=np.load(mean_file), channel_swap=(2, 1, 0), gpu=gpu_mode
        # )

        # read the synset and its meaning from synset file
        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            starttime = time.time()

            # here we call the script to classify image
            vgg_model_def_file = self.default_args['model_def_file']
            vgg_model_file = self.default_args['pretrained_model_file']
            vgg_mean_file = self.default_args['mean_file']

            print vgg_model_def_file
            print vgg_model_file
            print vgg_mean_file

            classify_cmd = '%s/vgg_classify.sh %s %s %s %s' % (VGG_CLASSIFY_DIR, vgg_model_def_file, \
                vgg_model_file, vgg_mean_file, image)
            print 'classify cmd %s' % classify_cmd
            # call the shell script, predite results written in 'predLabelsIdx.txt'
            subprocess.call(classify_cmd, shell=True)
            endtime = time.time()

            # now get the indices and scores from 'predLabelsIdx.txt'
            pred_results = np.loadtxt(os.path.join(VGG_CLASSIFY_DIR, 'predLabelsIdx.txt'))

            scores = pred_results
            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            ## for predict both sorted top-5 indices and scores in pred_results
            # res1 = np.int32(pred_results[:, 0]) # indices
            # res2 = np.float32(pred_results[:,1]) # scores
            #
            # indices = res1.tolist() # array to list
            # scores = res2.tolist()
            # predictions = self.labels[indices].tolist() # array to list

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            # for 1000 output scores
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]

            # for sorted top-5 output from vgg_classify
            # meta = [(p, '%.5f' % s) for s, p in zip(scores, predictions)]
            # logging.info('result: %s', str(meta))

            # Compute expected information gain
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']

            # sort the scores
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            logging.info('bet result: %s', str(bet_result))

            ## Here we rescale the predicted meta and bet_result
            # print 'old meta: ' + meta
            # print 'old bet: ' + bet_result
            #
            re_meta = rescale_score(meta)
            re_bet = rescale_score(bet_result)
            #
            # print 'new meta: ' + re_meta
            # print 'new bet: ' + re_bet
            # return True, re_meta, re_bet, '%.3f' % (endtime - starttime)

            return True, meta, bet_result, '%.3f' % (endtime - starttime)

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


def rescale_score(old_scores):
    """
        rescaleScore(old_scores) is a function to rescale the values of scores in old_scores.
        Here we use the simple rescale method: newscore = 1 / (1 + exp(-oldscore)).
    :param old_scores: a list of turbes, [(label1, oldscore1), (label2, oldscore2), ...]
    :return new_scores: a list of turbes, [(label1, newscore1), (label2, newscore2), ...]
    """
    new_scores = [];

    for (label, oldscore) in old_scores:
        newscore = 1 / (1 + np.exp(-float(oldscore)))
        new_scores.append((label, ('%.3f' % newscore)))

    return new_scores


# program starts from here
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
