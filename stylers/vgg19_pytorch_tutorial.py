# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
from torch import nn
from torch import optim

from torch.autograd import Variable
from torchvision import models, transforms
import matplotlib.pyplot as plt
from base import BaseStyler
import numpy as np
import copy

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

    def __init__(self,):
        self.model = models.vgg19(pretrained=True)

        cnn = copy.deepcopy(self.model.features)

        # Loss
        content_loss = []
        style_loss = []

        # A new model
        model = nn.Sequential()
        gram = GramMatrix()


        i = 1
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d):
                name = "conv_{}".format(i)
                model.add_module(name, layer)

            if name in content_layers_default:
                target = model()








    # def compute_features(self, img):
    #     '''
    #
    #     Compute intermediate activations
    #
    #     Parameters
    #     ----------
    #     img: torch.autograd.Variable
    #
    #     Returns
    #     -------
    #     features: list of torch.Tensor
    #         Feature maps
    #
    #     '''
    #     features = []
    #
    #     # Forward hook
    #     def save_features(self, inp, out):
    #         features.append(out)
    #
    #     for layer in self.model.features:
    #         if isinstance(layer, nn.ReLU):
    #             layer._forward_hooks = OrderedDict()
    #             layer.register_forward_hook(save_features)
    #
    #     self.model.features.forward(img)
    #
    #     return features

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
            total_loss = get_loss(random_features, original_features, style_features)
            total_loss.backward(retain_graph=True)
            return total_loss

        print "Optimizing..."
        # Optimization
        for i in range(8):
            loss_val = opt.step(loss_fun)
            print "Iter {}: {}".format(i, loss_val)



        #print total_loss

    def load_process(self, original_path, style_path):

        # Loading & transforming images
        original = load_image(original_path)
        style = load_image(style_path)

        self.process(original, style)

# ====================================================================== #
#                                 LOSS                                   #
# ====================================================================== #

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, *input):
        inp = input[0]
        self.loss = self.criterion(inp * self.weight, self.target)
        self.output = inp

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, *input):
        input = input[0]
        a, b, c, d = input.size()
        features = input.view(a*b, c*d)
        G = torch.mm(features, features.t())
        return G.div(a * b, c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, *input):
        inp = input[0]
        self.G = self.gram(inp)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


if __name__ == '__main__':

    styler = Vgg19Styler()

    IMG = "../dumps/tue.jpg"
    STYLE ="../dumps/vg.jpg"

    styler.load_process(IMG, STYLE)
