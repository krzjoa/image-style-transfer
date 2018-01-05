# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.utils import floatX
from utils import load
from scipy.ndimage import imread
from collections import OrderedDict

first = 64
second = 128
third = 256
fourth = 512

DUMP = "dumps/vgg19.pkl"


class VGG19Styler(object):

    def __init__(self):

        # Tensor
        self.input_image = T.tensor4('X', 'float64')
        # self.X_original = T.tensor4('X', 'float64')
        # self.X_styled = T.tensor4('X', 'float64')
        #
        self.output_labels = None
        self.mean_img = None
        self.model = self.build_net(self.input_image)

        # Image size
        self.image_width = None
        self.image_height = None

    def build_net(self, input_var):
        model = OrderedDict()

        model['input'] = lasagne.layers.InputLayer((None, 3,  224, 224),
                                          input_var=input_var)

        # 1. block
        model['conv1_1'] = lasagne.layers.Conv2DLayer(model['input'], first, 3, pad=1, flip_filters=False)
        model['conv1_2']  = lasagne.layers.Conv2DLayer(model['conv1_1'] , first, 3, pad=1, flip_filters=False)
        model['pool1'] = lasagne.layers.Pool2DLayer(model['conv1_2'] , 2, mode='average_exc_pad')

        # 2. block
        model['conv2_1'] = lasagne.layers.Conv2DLayer(model['pool1'], second, 3, pad=1, flip_filters=False)
        model['conv2_2'] = lasagne.layers.Conv2DLayer(model['conv2_1'], second, 3, pad=1, flip_filters=False)
        model['pool2'] = lasagne.layers.Pool2DLayer(model['conv2_2'], 2, mode='average_exc_pad')

        # 3, block
        model['conv3_1'] = lasagne.layers.Conv2DLayer(model['pool2'], third, 3, pad=1, flip_filters=False)
        model['conv3_2'] = lasagne.layers.Conv2DLayer(model['conv3_1'], third, 3, pad=1, flip_filters=False)
        model['conv3_3'] = lasagne.layers.Conv2DLayer(model['conv3_2'], third, 3, pad=1, flip_filters=False)
        model['conv3_4'] = lasagne.layers.Conv2DLayer(model['conv3_3'], third, 3, pad=1, flip_filters=False)
        model['pool3'] = lasagne.layers.Pool2DLayer(model['conv3_4'], 2, mode='average_exc_pad')

        model['conv4_1'] = lasagne.layers.Conv2DLayer(model['pool3'], fourth, 3, pad=1, flip_filters=False)
        model['conv4_2'] = lasagne.layers.Conv2DLayer(model['conv4_1'], fourth, 3, pad=1, flip_filters=False)
        model['conv4_3'] = lasagne.layers.Conv2DLayer(model['conv4_2'], fourth, 3, pad=1, flip_filters=False)
        model['conv4_4'] = lasagne.layers.Conv2DLayer(model['conv4_3'], fourth, 3, pad=1, flip_filters=False)
        model['pool4'] = lasagne.layers.Pool2DLayer(model['conv4_4'], 2, mode='average_exc_pad')

        model['conv5_1'] = lasagne.layers.Conv2DLayer(model['pool4'], fourth, 3, pad=1, flip_filters=False)
        model['conv5_2'] = lasagne.layers.Conv2DLayer(model['conv5_1'], fourth, 3, pad=1, flip_filters=False)
        model['conv5_3'] = lasagne.layers.Conv2DLayer(model['conv5_2'], fourth, 3, pad=1, flip_filters=False)
        model['conv5_4'] = lasagne.layers.Conv2DLayer(model['conv5_3'], fourth, 3, pad=1, flip_filters=False)
        model['pool5'] = lasagne.layers.Pool2DLayer(model['conv5_4'], 2, mode='average_exc_pad')

        # model = lasagne.layers.DenseLayer(model, num_units=4096)
        # model = lasagne.layers.DropoutLayer(model, p=0.5)
        # model = lasagne.layers.DenseLayer(model, num_units=4096)
        # model = lasagne.layers.DropoutLayer(model, p=0.5)
        # model = lasagne.layers.DenseLayer(model, num_units=1000, nonlinearity=None)
        # model = lasagne.layers.NonlinearityLayer(model, lasagne.nonlinearities.softmax)

        lasagne.layers.set_all_param_values(model['pool5'] , self.load_weigths())

        # Outputs
        self.outputs = lasagne.layers.get_output(model.values())

        return model

    def load_weigths(self):
        d = load(DUMP)
        self.output_labels = np.array(d['synset words'])
        self.mean_img = d['mean value']
        print "Mean value", self.mean_img
        return d['param values'][:32]

    def process(self, original, style):
        # Getting features (activations of each layer)
        original_image_features = self.compute_features(original)
        style_image_features = self.compute_features(style)

        # Assign image size
        self.image_width, self.image_height = img.shape

        # Generated image
        random_image = floatX(np.random.uniform(-128, 128, (1, 3, self.image_height, self.image_width)))
        random_image_features = self.compute_features(random_image)

    def compute_features(self, img):
        # Compute layer activations
        features = {
            k: theano.shared(output.eval({self.input_image: img}))
            for k, output in zip(self.model.keys(), self.outputs)
        }
        return features

    def forward(self, x):
        output = lasagne.layers.get_output(self.model, deterministic=True)
        process = theano.function([self.X], output)
        return process(x)

    def predict(self, x, n=1):
        out = self.forward(x)
        idx = np.argsort(out[0])[-1:-6:-1]
        return self.output_labels[idx]

    def prep_image(self, path):

        im = imread(path)
        # Resize so smallest dim = 256, preserving aspect ratio
        h, w, _ = im.shape
        # if h < w:
        #     im = skimage.transform.resize(im, (256, w * 256 / h), preserve_range=True)
        # else:
        #     im = skimage.transform.resize(im, (h * 256 / w, 256), preserve_range=True)

        # Central crop to 224x224
        h, w, _ = im.shape
        im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

        rawim = np.copy(im).astype('uint8')

        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

        # Convert to BGR
        im = im[::-1, :, :]

        im = im - self.mean_img[:, np.newaxis, np.newaxis]
        return rawim, lasagne.utils.floatX(im[np.newaxis])


MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
def preprocess(img):
    # img is (channels, height, width), values are 0-255
    img = img[::-1]  # switch to BGR
    img -= MEAN_VALUE[:, np.newaxis, np.newaxis]
    return img

# ============================================================= #
#                             UTILS                             #
# ============================================================= #

def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g

def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]

    loss = 0.5 * ((x-p)**2).sum()
    return loss

def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]

    A = gram_matrix(a)
    G = gram_matrix(x)

    N = a.shape[1]
    M = a.shape[2] * a.shape[3]

    loss = 1. / (4 * N ** 2 * M ** 2) * ((G - A) ** 2).sum()
    return loss


if __name__ == '__main__':
    net = VGG19Styler()

    # Loading image
    # img = imread("dumps/girl.jpg")
    # img = np.moveaxis(img, 2, 0).astype(np.float64)
    # img = preprocess(img)
    # img = np.expand_dims(img, 0).astype(np.float64)
    raw_img, img = net.prep_image("dumps/girl.jpg")
    #img = np.expand_dims(img, 0).astype(np.float64)

    #img = np.random.rand(1, 3, 224, 224).astype(np.float64)

    print img.shape

    net.process(img, img)

    #print net.forward(img).shape