import numpy as np

from modified_mnist.mnist_digit_writer import DigitWriter


class UniformDiscrete():
    """ samples x where min_ <= x < max_
    """
    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def sample(self):
        return int(np.random.uniform(self.min, self.max))


def shifted_mnist(observed_image):
    """ observes a blank image with an MNIST digit randomly drawn
        at an unknown location
    """
    # TODO: if this works, try changing the zoom of the image to
    #       properly test attention
    image_width = 100
    image_height = 100
    digit_width = DigitWriter.digit_width
    digit_height = DigitWriter.digit_height

    location_x_dist = UniformDiscrete(0, image_width-digit_width)
    location_y_dist = UniformDiscrete(0, image_height-digit_height)
    digit_dist = UniformDiscrete(0, 10)

    location_x = location_x_dist.sample()
    location_y = location_y_dist.sample()

    drawer = DigitWriter()
    digit = digit_dist.sample()
    digit_image = drawer.drawDigit(digit)

    full_image = np.zeros((image_height, image_width))
    full_image[location_y:location_y+digit_height,
               location_x:location_x+digit_width]\
        = digit_image

    """
    # can we now just observe the image with noise added to each pixel?

    mvn(full_image.reshape((-1)),
        np.identity(image_width*image_height)).observe(observed_image)
    """
    return (full_image, digit, [location_x, location_y])


if __name__ == '__main__':
    from PIL import Image
    img = Image.fromarray(shifted_mnist(None))
    img.show()
