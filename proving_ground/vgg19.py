# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import scipy.optimize

from torch.autograd import Variable
from torchvision import models, transforms
import matplotlib.pyplot as plt
from stylers.base import BaseStyler
import numpy as np


# ====================================================================== #
#                               PARAMETERS                               #
# ====================================================================== #

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# ====================================================================== #
#                               LOADING IMAGE                            #
# ====================================================================== #


def load_image(path):
    img = plt.imread(path)
    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        ]
    )

    img = transformer(img)
    return img.unsqueeze(0)


class Vgg19Styler(BaseStyler):

    def __init__(self):
        self.model = models.vgg19(pretrained=True)
        #self.model.cuda()

    def compute_features(self, img):
        '''

        Compute intermediate activations

        Parameters
        ----------
        img: torch.autograd.Variable

        Returns
        -------
        features: list of torch.Tensor
            Feature maps

        '''
        features = []

        # Forward hook
        def save_features(self, inp, out):
            features.append(out)

        for layer in self.model.features:
            if isinstance(layer, nn.ReLU):
                layer._forward_hooks = OrderedDict()
                layer.register_forward_hook(save_features)

        self.model.features.forward(img)
        return features

    def process(self, original, style):

        original = Variable(original)
        style = Variable(style)

        # Random image
        random_image = torch.rand(*list(original.shape))
        random_image = Variable(random_image, requires_grad=True)

        # Computing features
        print "Computing features..."
        original_features = self.compute_features(original)
        style_features = self.compute_features(style)
        random_features = self.compute_features(random_image)

        opt = optim.LBFGS([random_image])

        # Loss function
        def loss_fun():
            opt.zero_grad()
            self.model.features(random_image)
            total_loss = get_loss(random_features, original_features, style_features)
            total_loss.backward(retain_graph=True)
            return total_loss

        print "Optimizing..."
        # Optimization
        for i in range(8):
            loss_val = opt.step(loss_fun)
            print "Iter {}: {}".format(i, loss_val)

    def load_process(self, original_path, style_path):

        # Loading & transforming images
        original = load_image(original_path)
        style = load_image(style_path)

        self.process(original, style)

# ================================================== #
#                           LOSS                     #
# ================================================== #


def get_loss(random_fm_list, content_fm_list, style_fm_list, alpha=0.001):

    # print "Getting loss..."

    total_loss = []

    for rf, cf, sf in zip(random_fm_list, content_fm_list, style_fm_list):
        total_loss.append(alpha * style_loss(rf, sf))
        total_loss.append(alpha * content_loss(rf, cf))

    return sum(total_loss)


def content_loss(radom_fm, content_fm):
    '''

    Content loss by Gatys et al.

    Parameters
    ----------
    radom_fm: torch.Tensor
        Feature map for random image
    content_fm:
        Feature map for style image

    Returns
    -------

    '''

    return 0.5 * ((radom_fm - content_fm) ** 2).sum()


def style_loss(radom_fm, style_fm):
    '''

    Style loss by Gatys et al.

    Parameters
    ----------
    radom_fm: torch.Tensor
        Feature map for random image
    style_fm:
        Feature map for style image

    Returns
    -------

    '''

    a = gram_matrix(radom_fm)
    g = gram_matrix(style_fm)

    N = a.shape[1]
    M = a.shape[2] * a.shape[3]

    loss = 1./(4 * N**2 * M**2) * ((g- a)**2).sum()
    return loss


def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a*b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a* b * c *d)



if __name__ == '__main__':

    styler = Vgg19Styler()

    IMG = "../dumps/tue.jpg"
    STYLE ="../dumps/vg.jpg"

    styler.load_process(IMG, STYLE)





