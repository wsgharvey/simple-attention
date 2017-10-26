import os
import numpy as np
from mnist import MNIST
mndata = MNIST(os.getenv("HOME")+"/.modified_mnist")


class DigitWriter(object):
    """ Provides function to give MNIST image containing a specified digit
    """
    digit_width, digit_height = (28, 28)

    def __init__(self):
        self.mnist_images = {digit: [] for digit in range(10)}

        images, labels = mndata.load_training()

        for image, label in zip(images, labels):
            self.mnist_images[label].append(image)

    def drawDigit(self, digit):
        """ Returns an image containing the digit from MNIST as a numpy array
        """
        random_index = np.random.randint(len(self.mnist_images[digit]))

        return np.array(
                        self.mnist_images[digit][random_index]
                        ).reshape((28, 28))
