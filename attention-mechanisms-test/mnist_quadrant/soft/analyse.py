
import PIL
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from numbers import Number
import numpy as np


def analysePredictions(net, data_generator, n_data=np.inf):
    """ produces a report on the current performance of the neural net
        image of data + attention filters + bar chart of prediction certainty
    """
    line_width = 70
    full_image_x = 0
    low_res_x = 60
    attention_x = 120
    predictions_x = 180
    end = 280

    images, targets = next(data_generator)
    proposals = net(images).data.numpy()
    attention_weights_list = net.getAttentionSummary().data.numpy()

    full_images = images.fullView().data.numpy()
    low_res_images = images.lowResView().data.numpy()

    canvas = PIL.Image.new('L', (end,
                                 min(n_data, len(proposals) * line_width)),
                           100)

    datum = 0
    while datum < n_data:
        try:
            proposal = proposals[datum]
            attention_weights = attention_weights_list[0]
            full_image = PIL.Image.fromarray(full_images[datum])

            # expand low res. to sam size as full res.
            low_res_image = PIL.Image.fromarray(
                                np.kron(low_res_images[datum],
                                        np.ones((4, 4))))
        except IndexError:
            break

        canvas.paste(full_image,
                     (full_image_x, datum*line_width))
        canvas.paste(low_res_image,
                     (low_res_x, datum*line_width))

        # Create graphic to show attention weights
        graphic = np.zeros((56, 56))
        locations = {0: (0, 0),
                     1: (0, 28),
                     2: (28, 0),
                     3: (28, 28)}
        for region in locations:
            weight = attention_weights[region]
            loc_x, loc_y = locations[region]
            graphic[loc_y:loc_y+28, loc_x:loc_x+28] =\
                np.ones((28, 28)) * 255 * weight
        graphic = PIL.Image.fromarray(graphic)
        canvas.paste(graphic,
                     (attention_x, datum*line_width))

        # Now create graphic to show its guess for the digit
        bar_width = 10
        max_bar_height = 56
        graphic = PIL.Image.new('L', (bar_width*10, max_bar_height))
        for digit in range(10):
            bar = np.ones((max_bar_height, bar_width))*255
            bar_distance_from_max = int(max_bar_height
                                        * (1 - np.exp(proposal[digit])))
            bar[bar_distance_from_max:, :] *= 0
            bar = PIL.Image.fromarray(bar)
            graphic.paste(bar,
                          (digit*bar_width, 0))
        canvas.paste(graphic,
                     (predictions_x, datum*line_width))

        datum += 1
    return canvas


class trackStat(object):
    """ some sort of object that can have the loss added at each iteration and
        then do a plot at the end
    """
    def __init__(self, stat_name="Loss"):
        self.stats = []
        self.n_iterations_list = []
        self.stat_name = stat_name

    def add(self, stat, n_iterations):
        """ adds stat and n_iterations to lists for future use
        - stat can be 1x1 Variable or Tensor or Number
        """
        assert isinstance(n_iterations, int)

        if isinstance(stat, Variable):
            stat = stat.data
        if isinstance(stat, torch.Tensor):
            stat = stat.view(1)[0]
        assert isinstance(stat, Number)

        self.stats.append(stat)
        self.n_iterations_list.append(n_iterations)

    def plot(self, save_file=None):
        """ does some find of plot
        """
        plt.plot(self.n_iterations_list, self.stats)
        plt.xlabel("Iterations")
        plt.ylabel(self.stat_name)
        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file)
