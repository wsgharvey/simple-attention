import torch
import torch.nn as nn
from torch.autograd import Variable

from ..base import Images
from modified_mnist.three_mnist import ThreeMNIST


class TripleMNISTImages(Images):
    def __init__(self, image_tensor, cuda=False):
        """ takes in n_images x 140 x 28 Tensor
        """
        self.cuda = cuda

        self.n_images = len(image_tensor)

        self.full_images = image_tensor

        reduceRes = torch.nn.AvgPool2d(10, stride=9)

        self.low_res = reduceRes(
                        self.full_images.view(self.n_images, 1, 28, 140))\
            .view(self.n_images, 3, 15)

    def lowResView(self):
        if self.cuda:
            return self.low_res.cuda()
        else:
            return self.low_res

    def focusView(self, locations, loc_type="discrete"):
        """ if loc_type is "discrete", location is integer 0 ... 14
        """
        assert len(locations) == self.n_images

        for loc in locations:
            assert loc in range(15)

        focusedViews = Variable(torch.zeros(self.n_images, 28, 28))

        for img_no in range(self.n_images):
            location = locations[img_no]
            locations[img_no] = self.full_images[img_no,
                                                 :,
                                                 8*location:8*location+28]
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
                               cuda=False):
    """ yields tuples of images and targets """
    generator = ThreeMNIST(single_target=single_target)
    while True:
        images, targets = generator.generate(n_images=batch_size,
                                             noise_SD=noise_SD)
        images = TripleMNISTImages(images, cuda=cuda)
        yield images, targets
