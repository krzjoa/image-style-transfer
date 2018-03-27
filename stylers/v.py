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
runs = 200

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

    def compute_features(self, content_img, style_img, style_weight=1000, content_weight=1):
        cnn = copy.deepcopy(self.model.features)

        # Loss
        content_loss_all = []
        style_loss_all = []

        # A new model
        model = nn.Sequential()
        gram = GramMatrix()

        i = 1
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ReLU):
                name = "conv_{}".format(i) if isinstance(layer, nn.Conv2d) else "relu_{}".format(i)
                model.add_module(name, layer)

                if name in content_layers_default:
                    target = model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    model.add_module("content_loss_{}".format(i), content_loss)
                    content_loss_all.append(content_loss)

                if name in style_layers_default:
                    target_feature = model(style_img).clone()
                    target_feature_gram = gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    model.add_module("style_loss_{}".format(i), style_loss)
                    content_loss_all.append(style_loss)

                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)
                model.add_module(name, layer)

        return model, style_loss_all, content_loss_all


    def process(self, content, style):

        content = Variable(content)
        style = Variable(style)

        # Random image
        random_image = torch.rand(*list(content.shape))
        random_image_param = nn.Parameter(random_image)
        #random_image = Variable(random_image, requires_grad=True)

        model, style_losses, content_losses = self.compute_features(content, style)

        opt = optim.LBFGS([random_image_param])

        print "Optimizing..."
        # Optimization

        for i in range(runs):
        # Loss function
            def closure():
                random_image_param.data.clamp_(0, 1)

                opt.zero_grad()
                self.model(random_image_param)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.backward()
                for cl in content_losses:
                    content_score += cl.backward()

                if i % 50 == 0:
                    print("run {}:".format(runs))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.data[0], content_score.data[0]))
                    print()
                return style_score + content_score

            loss_val = opt.step(closure)
            print "Iter {}: {}".format(i, loss_val)

        random_image_param.data.clamp_(0, 1)
        return random_image_param.data

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

    def forward(self, input):
        #inp = input[0]
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):

    def forward(self, input):
        #input = input[0]
        a, b, c, d = input.size()
        features = input.view(a*b, c*d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        #inp = input[0]
        self.output = input.clone()
        self.G = self.gram(input)
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
