import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from ..base import Images
from modified_mnist.three_mnist import ThreeMNIST


class TripleMNISTImages(Images):
    def __init__(self, image_tensor, cuda=False, n_locations=15):
        """ takes in n_images x 140 x 28 Tensor
        """
        self.cuda = cuda

        self.n_images = len(image_tensor)

        self.full_images = image_tensor

        reduceRes = torch.nn.AvgPool2d(10, stride=9)

        self.low_res = reduceRes(
                        self.full_images.view(self.n_images, 1, 28, 140))\
            .view(self.n_images, 3, 15)

        self.n_locations = n_locations
        self.discrete_locations = [int(l) for l in np.arange(0,
                                                             140,
                                                             (140-28)/(n_locations-1))]

    def lowResView(self):
        if self.cuda:
            return self.low_res.cuda()
        else:
            return self.low_res

    def focusView(self, desired_locations, loc_type="discrete"):
        """ if loc_type is "discrete", location is integer 0 ... self.n_locations
        """
        assert len(desired_locations) == self.n_images

        for loc in desired_locations:
            assert loc in range(self.n_locations)

        focusedViews = Variable(torch.zeros(self.n_images, 28, 28))

        for img_no in range(self.n_images):
            location = desired_locations[img_no]
            loc_x = self.discrete_locations[location]
            desired_locations[img_no] = self.full_images[img_no,
                                                         :,
                                                         loc_x:loc_x+28]
        if self.cuda:
            return focusedViews.cuda()
        else:
            return focusedViews

    def fullView(self):
        if self.cuda:
            return self.full_images.cuda()
        else:
            return self.full_images

    def nImages(self):
        return self.n_images


def TripleMNISTImagesGenerator(batch_size=1,
                               noise_SD=0,
                               single_target=False,
                               cuda=False,
                               n_locations=15):
    """ yields tuples of images and targets """
    generator = ThreeMNIST(single_target=single_target)
    while True:
        images, targets = generator.generate(n_images=batch_size,
                                             noise_SD=noise_SD)
        images = TripleMNISTImages(images, cuda=cuda, n_locations=n_locations)
        if cuda:
            targets = targets.cuda()
        yield images, targets
