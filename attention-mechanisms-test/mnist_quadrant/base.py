""" Defining overall neural net structure for test """
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Images(object):
    """ base class for a stack of images
        allows low res. view and focused view to be extracted
    """
    @abc.abstractmethod
    def lowResView(self):
        """ return low resolution views of entire images
        """

    @abc.abstractmethod
    def focusView(self, region):
        """ return high resolution views of particular regions
            - region could be an list of ints if categorical or
              list of coordinates etc.
        """

    @abc.abstractmethod
    def fullView(self):
        """ returns high resolution view of entire image
        """

    @abc.abstractmethod
    def nImages(self):
        """ return number of images
        """


class AttentionBox(nn.Module):
    """ box which receives low-res input and outputs embedding of
        focused region
    """
    @abc.abstractmethod
    def forward(images):
        """ outputs focused embedding
        """

    @abc.abstractmethod
    def getAttentionSummary(self):
        """ returns an object to represent the attention locations used by the
            network in the previous call of forward, or None if forward has not
            been called
        """


class LowResEmbedder(nn.Module):
    """ takes in low-res input and outputs embedding of low-res image
    """
    @abc.abstractmethod
    def forward(images):
        """ outputs embedding of images.LowResView()
        """


class CoreAndProposalLayer(nn.Module):
    """ combines function of LSTM and proposal layer to take inputs from the
        AttentionBox and LowResEmbedder and output proposal dist. params
    """
    @abc.abstractmethod
    def forward(low_res_embedding, focus_embedding):
        """ outputs proposal params
            eg vector of log probabilities for MNIST classification
        """


class FullNet(nn.Module):
    """ network constructed from AttentionBox, LowResEmbedder and
        CoreAndProposalLayer
    """
    def __init__(self, attention_box=0,
                 low_res_embedder=0,
                 core_proposal_layer=0):
        assert isinstance(attention_box, AttentionBox)
        assert isinstance(low_res_embedder, LowResEmbedder)
        assert isinstance(core_proposal_layer, CoreAndProposalLayer)

        super(FullNet, self).__init__()

        self.attention_box = attention_box
        self.low_res_embedder = low_res_embedder
        self.core_proposal_layer = core_proposal_layer

    def forward(self, images):
        assert isinstance(images, Images)

        focus_embedding = self.attention_box(images)
        low_res_embedding = self.low_res_embedder(images)

        proposal_dist = self.core_proposal_layer(low_res_embedding,
                                                 focus_embedding)
        return proposal_dist

    def getAttentionSummary(self):
        """ returns an object to represent the attention locations used by the
            network in the previous call of forward, or None if forward has not
            been called
        """
        return self.attention_box.getAttentionSummary()
