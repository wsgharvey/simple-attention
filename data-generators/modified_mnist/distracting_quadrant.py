import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from modified_mnist.mnist_digit_writer import DigitWriter

import time


class QuadrantMNIST(object):
    """ provides function to generate quadranted MNISTQuadrantImages
    """
    def __init__(self):
        self.drawer = DigitWriter()

    def generate(self,
                 n_images=1,
                 noise_SD=0,
                 n_distractors=5):
        """ return a 56x56 tensor with an MNIST digit in one of the 4 corner
        """

        mnist_width = 28
        mnist_height = 28

        full_images = torch.zeros(n_images, 2*mnist_height, 2*mnist_width)
        digits = Variable(torch.zeros(n_images).type(torch.LongTensor))

        for image_no in range(n_images):
            corner_no = np.random.randint(4)
            digit = np.random.randint(10)
            digits[image_no] = digit

            mnist_image = torch.Tensor(
                            self.drawer.drawDigit(digit).astype(float))

            if corner_no == 0:
                start_x = 0
                start_y = 0
            elif corner_no == 1:
                start_x = 0
                start_y = mnist_height
            elif corner_no == 2:
                start_x = mnist_width
                start_y = 0
            elif corner_no == 3:
                start_x = mnist_width
                start_y = mnist_height

            full_images[image_no,
                        start_y:start_y+mnist_height,
                        start_x:start_x+mnist_width] = mnist_image

            shrinker = nn.MaxPool2d(2, 2)  # will set it to size 14x14
            for _ in range(n_distractors):
                # use scaled down MNIST digit as some structured noise
                distractor = torch.Tensor(
                                self.drawer.drawDigit(np.random.randint(10))
                                    .astype(float))
                distractor = shrinker(distractor.view(1, 28, 28)).view(14, 14)
                loc_x = np.random.randint(56-14)
                loc_y = np.random.randint(56-14)
                full_images[image_no,
                            loc_y:loc_y+14,
                            loc_x:loc_x+14] += distractor.data

        full_images = Variable(full_images)
        if noise_SD > 0:
            full_images += noise_SD * Variable(torch.randn(full_images.size()))
            full_images = F.relu(full_images)
        return full_images, digits


if __name__ == '__main__':
    from PIL import Image
    quadrant_mnist = QuadrantNIST()
    img = Image.fromarray(quadrant_mnist.generate(noise_SD=2).data.numpy()[0])
    img.show()
