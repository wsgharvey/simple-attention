import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from ..base import LowResEmbedder, AttentionBox, CoreAndProposalLayer


class TripleMNISTLowResEmbedder(LowResEmbedder):
    def __init__(self):
        super(TripleMNISTLowResEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.conv2 = nn.Conv1d(20, 10, 4)
        self.fcn1 = nn.Linear(100, 40)
        self.fcn2 = nn.Linear(40, 20)

    def forward(self, images):
        x = images.lowResView()
        x = x.view(-1, 1, 3, 15)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 20, 13)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 100)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        return x


class TripleMNISTCoreAndProposalLayer(CoreAndProposalLayer):
    def __init__(self):
        super(TripleMNISTCoreAndProposalLayer, self).__init__()
        self.fcn1 = nn.Linear(40, 40)
        self.fcn2 = nn.Linear(40, 20)
        self.fcn3 = nn.Linear(20, 10)

    def forward(self, low_res_embedding, focus_embedding):
        x = torch.cat([low_res_embedding, focus_embedding], 1)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = F.relu(self.fcn3(x))
        proposal_dist = F.log_softmax(x)
        return proposal_dist


class TripleMNISTSoftAttentionBox(AttentionBox):
    def __init__(self, weight_layer_type="fcn", attention_type="soft"):
        super(TripleMNISTSoftAttentionBox, self).__init__()
        if weight_layer_type == "fcn":
            self.attention_weights_layer = TripleMNISTAttentionWeightsLayer()
        elif weight_layer_type == "conv":
            self.attention_weights_layer = TripleMNISTConvAttentionWeightsLayer()
        else:
            raise Exception("{} not a valid type".format(weight_layer_type))
        self.focus_embedder = TripleMNISTFocusEmbedder()
        self.prev_attention_weights = None
        self.n_locations = 13
        self.attention_type = attention_type

    def forward(self, images):
        low_res_view = images.lowResView()
        attention_weights = self.attention_weights_layer(low_res_view)

        self.most_recent_attention_weights = attention_weights

        if self.attention_type == "soft":
            # add a weighted embedding of each view to the full embedding
            for location in range(self.n_locations):
                high_res_images = images.focusView([location]*images.nImages())
                high_res_images = high_res_images.view(-1, 1, 28, 28)
                local_focus_embeddings = self.focus_embedder(high_res_images)

                local_attention_weights = attention_weights[:, location]
                if location == 0:
                    focus_embedding = local_focus_embeddings * 0
                for img_no in range(images.nImages()):  # TODO: check this
                    focus_embedding[img_no] = focus_embedding[img_no]\
                                                    + local_focus_embeddings[img_no]\
                                                    * local_attention_weights[img_no]
        elif self.attention_type == "hard":
            n_samples = 5

            # do some sneaky stuff while hiding from autograd:
            #   (get the samples * 1/q)
            np_attention_weights = attention_weights.data.numpy()
            attention_choices = [np.random.multinomial(n_samples, img_probs)/n_samples for img_probs in np_attention_weights]
            attention_choices = np.divide(attention_choices, np_attention_weights)
            attention_choices = Variable(torch.from_numpy(np.array(attention_choices))).type(torch.FloatTensor)

            # we've done our job and can let autograd watch us again
            # restore to original samples with weight of 1 (but variable)
            attention_choices = torch.mul(attention_choices, attention_weights)
            for location in range(self.n_locations):
                high_res_images = images.focusView([location]*images.nImages())
                high_res_images = high_res_images.view(-1, 1, 28, 28)
                local_focus_embeddings = self.focus_embedder(high_res_images)

                local_attention_choices = attention_choices[:, location]
                if location == 0:
                    focus_embedding = local_focus_embeddings * 0
                for img_no in range(images.nImages()):  # TODO: check this
                    focus_embedding[img_no] = focus_embedding[img_no]\
                                                    + local_focus_embeddings[img_no]\
                                                    * local_attention_choices[img_no]

        else:
            raise Exception("{} not a valid attention type".format(self.attention_type))
        return focus_embedding

    def getAttentionSummary(self):
        """ returns Variable of attention weights
        """
        return self.most_recent_attention_weights


class TripleMNISTHardAttentionBox(AttentionBox):
    def __init__(self,  weight_layer_type="fcn"):
        super(TripleMNISTHardAttentionBox, self).__init__()
        ...


class TripleMNISTAttentionWeightsLayer(nn.Module):
    def __init__(self):
        super(TripleMNISTAttentionWeightsLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.conv2 = nn.Conv1d(20, 10, 4)
        self.fcn1 = nn.Linear(100, 40)
        self.fcn2 = nn.Linear(40, 13)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 1, 3, 15)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 20, 13)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 100)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        weights = self.softmax(x)
        self.previous_weights = weights
        return weights


class TripleMNISTConvAttentionWeightsLayer(nn.Module):
    def __init__(self):
        super(TripleMNISTConvAttentionWeightsLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=0)
        self.conv3 = nn.Conv1d(10, 1, 5, padding=2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 1, 3, 15)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10, 13)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 13)
        weights = self.softmax(x)
        self.previous_weights = weights
        return weights


class TripleMNISTFocusEmbedder(nn.Module):
    def __init__(self):
        super(TripleMNISTFocusEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(20, 10, 3)
        self.fcn1 = nn.Linear(90, 40)
        self.fcn2 = nn.Linear(40, 20)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 90)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        return x
