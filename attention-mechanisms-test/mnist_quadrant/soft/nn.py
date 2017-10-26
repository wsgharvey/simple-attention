import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from ..base import LowResEmbedder, AttentionBox, CoreAndProposalLayer


class MNISTQuadrantLowResEmbedder(LowResEmbedder):
    def __init__(self):
        super(MNISTQuadrantLowResEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.fcn1 = nn.Linear(160, 40)
        self.fcn2 = nn.Linear(40, 20)

    def forward(self, images):
        x = images.lowResView()
        x = x.view(-1, 1, 14, 14)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 160)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        return x


class MNISTQuadrantCoreAndProposalLayer(CoreAndProposalLayer):
    def __init__(self):
        super(MNISTQuadrantCoreAndProposalLayer, self).__init__()
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


class MNISTQuadrantAttentionBox(AttentionBox):
    def __init__(self):
        super(MNISTQuadrantAttentionBox, self).__init__()
        self.attention_weights_layer = MNISTQuadrantAttentionWeightsLayer()
        self.focus_embedder = MNISTQuadrantFocusEmbedder()
        self.prev_attention_weights = None

    def forward(self, images):
        low_res_view = images.lowResView()
        attention_weights = self.attention_weights_layer(low_res_view)

        self.most_recent_attention_weights = attention_weights

        focus_embedding = Variable(torch.zeros(images.nImages(), 20))

        # add a weighted embedding of each view to the full embedding
        for location in [0, 1, 2, 3]:
            high_res_images = images.focusView([location]*images.nImages())
            high_res_images = high_res_images.view(-1, 1, 28, 28)
            local_focus_embeddings = self.focus_embedder(high_res_images)

            local_attention_weights = attention_weights[:, location]
            for img_no in range(images.nImages()):  # TODO: check this
                focus_embedding[img_no] = focus_embedding[img_no]\
                                            + local_focus_embeddings[img_no]\
                                            * local_attention_weights[img_no]
        return focus_embedding

    def getAttentionSummary(self):
        """ returns Variable of attention weights
        """
        return self.most_recent_attention_weights


class MNISTQuadrantAttentionWeightsLayer(nn.Module):
    def __init__(self):
        super(MNISTQuadrantAttentionWeightsLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 5, 3)
        self.fcn1 = nn.Linear(80, 16)
        self.fcn2 = nn.Linear(16, 4)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 1, 14, 14)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 80)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        weights = self.softmax(x)
        self.previous_weights = weights
        return weights


class MNISTQuadrantFocusEmbedder(nn.Module):
    def __init__(self):
        super(MNISTQuadrantFocusEmbedder, self).__init__()
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
