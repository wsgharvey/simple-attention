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
            self.attention_weights_layer = TripleMNISTConvDiscreteAttentionWeightsLayer()
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
        # _, best_locations = torch.max(attention_weights, 1)
        # best_locations = list(best_locations.data.numpy())
        # self.most_recent_best_images = images.focusView(best_locations)

        if self.attention_type == "soft":
            # add a weighted embedding of each view to the full embedding
            for location in range(self.n_locations):
                high_res_images = images.focusView([location]*images.nImages(),
                                                   loc_type == "discrete")
                high_res_images = high_res_images.view(-1, 1, 28, 28)
                local_focus_embeddings = self.focus_embedder(high_res_images)

                local_attention_weights = attention_weights[:, location]
                if location == 0:
                    focus_embedding = local_focus_embeddings * 0
                for img_no in range(images.nImages()):  # TODO: check this
                    focus_embedding[img_no] = focus_embedding[img_no]\
                                                    + local_focus_embeddings[img_no]\
                                                    * local_attention_weights[img_no]  # fuck it up a bit to see what happens
        elif self.attention_type == "hard":
            n_samples = 5

            # do some sneaky stuff while hiding from autograd:
            #   (get the samples * 1/q)
            np_attention_weights = attention_weights.data.numpy()
            try:
                attention_choices = [np.random.multinomial(n_samples, img_probs/(sum(img_probs)+1e-4))/n_samples for img_probs in np_attention_weights]
            except:
                for img_probs in np_attention_weights:
                    print(img_probs/sum(img_probs), sum(img_probs))
                raise Exception
            divided_choices = []
            for choices, weights in zip(attention_choices, np_attention_weights):
                divided_choice = []
                for choice, weight in zip(choices, weights):
                    if choice == 0:
                        divided_choice.append(choice)
                    else:
                        divided_choice.append(choice/weight)
                divided_choices.append(divided_choice)
            attention_choices = np.array(divided_choices)
            attention_choices = Variable(torch.from_numpy(np.array(attention_choices))).type(torch.FloatTensor)

            # we've done our job and can let autograd watch us again
            # restore to original samples with weight of 1 (but variable)
            attention_choices = torch.mul(attention_choices, attention_weights)
            for location in range(self.n_locations):
                high_res_images = images.focusView([location]*images.nImages(),
                                                   loc_type="discrete")
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


class TripleMNISTConvDiscreteAttentionWeightsLayer(nn.Module):
    """ returns coordinate in range 0 - 1
    """
    def __init__(self):
        super(TripleMNISTConvDiscreteAttentionWeightsLayer, self).__init__()
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


class TripleMNISTConvContinuousAttentionWeightsLayer(nn.Module):
    """ returns categorical distribution over 13 views
    """
    def __init__(self):
        super(TripleMNISTConvContinuousAttentionWeightsLayer, self).__init__()
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

        # transform discrete weights into coordinate in range 0 - 1
        transformer = Variable(torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
        coord = torch.dot(weights, transformer)

        self.previous_weights = coord
        return coord


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
