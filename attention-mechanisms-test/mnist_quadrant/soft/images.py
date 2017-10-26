import torch
import torch.nn as nn
from torch.autograd import Variable

from ..base import Images
from modified_mnist.distracting_quadrant import QuadrantMNIST


class MNISTQuadrantImages(Images):
    def __init__(self, image_tensor):
        """ takes in n_images x 56 x 56 Tensor
        """
        self.n_images = image_tensor.size()[0]

        self.full_images = image_tensor

        reduceRes = torch.nn.AvgPool2d(4, 4)
        self.low_res = reduceRes(
                        self.full_images.view(self.n_images, 1, 56, 56))\
            .view(self.n_images, 14, 14)

    def lowResView(self):
        return self.low_res

    def focusView(self, locations):
        """ location is integer 0, 1, 2, 3
        """
        assert len(locations) == self.n_images

        for loc in locations:
            assert loc in [0, 1, 2, 3]

        focusedViews = Variable(torch.zeros(self.n_images, 28, 28))

        for img_no in range(self.n_images):
            if locations[img_no] == 0:
                focusedViews[img_no] = self.full_images[img_no, 0:28, 0:28]
            elif locations[img_no] == 1:
                focusedViews[img_no] = self.full_images[img_no, 0:28, 28:56]
            elif locations[img_no] == 2:
                focusedViews[img_no] = self.full_images[img_no, 28:56, 0:28]
            elif locations[img_no] == 3:
                focusedViews[img_no] = self.full_images[img_no, 28:56, 28:56]
        return focusedViews

    def fullView(self):
        return self.full_images

    def nImages(self):
        return self.n_images


def MNISTQuadrantImagesGenerator(batch_size=1, noise_SD=0):
    """ yields tuples of images and targets """
    generator = QuadrantMNIST()
    while True:
        images, targets = generator.generate(n_images=batch_size,
                                             noise_SD=noise_SD)
        images = MNISTQuadrantImages(images)
        yield images, targets
