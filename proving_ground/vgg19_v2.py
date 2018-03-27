# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
from torch import nn
from torch import optim

from torch.autograd import Variable
from torchvision import models, transforms
import matplotlib.pyplot as plt
from stylers.base import BaseStyler
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

    def get_style_model_and_losses(self, cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        model = nn.Sequential()  # the new Sequential module network
        gram = GramMatrix()  # we need a gram module in order to compute style targets

        # move these modules to the GPU if possible:
        # if use_cuda:
        #     model = model.cuda()
        #     gram = gram.cuda()
        #
        i = 1
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = model(style_img).clone()
                    target_feature_gram = gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                model.add_module(name, layer)

                if name in content_layers:
                    # add content loss:
                    target = model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # add style loss:
                    target_feature = model(style_img).clone()
                    target_feature_gram = gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                model.add_module(name, layer)  # ***

        return model, style_losses, content_losses

    def get_input_param_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        input_param = nn.Parameter(input_img.data)
        optimizer = optim.LBFGS([input_param])
        return input_param, optimizer


    def process(self, content_img, style_img, style_weight=1000, content_weight=1):

        content_img = Variable(content_img)
        style_img = Variable(style_img)

        input_img = Variable(torch.randn(content_img.size()))#.type(dtype)

        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(self.model.features,
                                                                         style_img, content_img, style_weight,
                                                                         content_weight)
        input_param, optimizer = self.get_input_param_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= runs:

            def closure():
                # correct the values of updated input image
                input_param.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_param)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.backward()
                for cl in content_losses:
                    content_score += cl.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.data[0], content_score.data[0]))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_param.data.clamp_(0, 1)

        return input_param.data


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
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
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
