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
    def __init__(self, type="fcn"):
        super(TripleMNISTSoftAttentionBox, self).__init__()
        self.attention_weights_layer = TripleMNISTAttentionWeightsLayer()
        self.focus_embedder = TripleMNISTFocusEmbedder()
        self.prev_attention_weights = None
        self.n_locations = 13

    def forward(self, images):
        low_res_view = images.lowResView()
        attention_weights = self.attention_weights_layer(low_res_view)

        self.most_recent_attention_weights = attention_weights 

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
        return focus_embedding

    def getAttentionSummary(self):
        """ returns Variable of attention weights
        """
        return self.most_recent_attention_weights


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
        super(TripleMNISTAttentionWeightsLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=0)
        self.conv3 = nn.Conv1d(20, 20, 3)
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
