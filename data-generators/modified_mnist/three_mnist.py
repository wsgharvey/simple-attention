import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from modified_mnist.mnist_digit_writer import DigitWriter

import time


class ThreeMNIST(object):
    """ provides function to generate quadranted images containing three
        consecutive MNIST digits
    """
    def __init__(self,
                 single_target=False,
                 give_locations=False):
        self.drawer = DigitWriter()
        self.single_target = single_target
        self.give_locations = give_locations

    def generate(self,
                 n_images=1,
                 noise_SD=0):
        """ return a 28x140 tensor with an MNIST digit in one of the 4 corner
             - if single_target, will only return first digit as the target
        """

        mnist_width = 28
        mnist_height = 28

        n_digits = 3
        digits_wide = 5

        full_images = Variable(
                        torch.zeros(n_images,
                                    mnist_height,
                                    digits_wide*mnist_width))
        full_digits = Variable(torch.zeros(n_images,
                                           n_digits).type(torch.LongTensor))

        locations = []
        for image_no in range(n_images):
            digits = np.random.randint(10, size=(3))
            digits_start = np.random.randint((digits_wide-n_digits)
                                             * mnist_width)
            locations.append(digits_start)
            full_digits[image_no] = torch.from_numpy(digits)

            for index, digit in enumerate(digits):
                full_images[image_no, :,
                            digits_start+index*mnist_width:
                                digits_start+(index+1)*mnist_width] =\
                    torch.Tensor(self.drawer.drawDigit(digit).astype(float))

        if noise_SD > 0:
            full_images += noise_SD * Variable(torch.randn(full_images.size()))
            full_images = F.relu(full_images)

        if self.single_target:
            digit_info = full_digits[:, 0]
        else:
            digit_info = full_digits

        if self.give_locations:
            locations = Variable(torch.Tensor(locations)).type('torch.FloatTensor')
            return full_images, digit_info, locations
        else:
            return full_images, digit_info



if __name__ == '__main__':
    from PIL import Image
    gen = ThreeMNIST()
    for i in range(10):
        img = Image.fromarray(gen.generate(noise_SD=2)[0].data.numpy()[0], mode='F')
        img = img.convert('RGB')
        img.save("/home/will/junk/img_{}.png".format(i))
