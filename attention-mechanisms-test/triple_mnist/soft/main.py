import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .images import TripleMNISTImagesGenerator,\
                    give_ideal_locations
from .nn import TripleMNISTLowResEmbedder,\
               TripleMNISTSoftAttentionBox,\
               TripleMNISTCoreAndProposalLayer
from ..base import FullNet
from .analyse import analysePredictions, trackStat


def run(attention_layer_type="fcn",
        step_size=1e-3,
        batch_size=10,
        noise_SD=0,
        iterations=10,
        graphics_path=None,
        cuda=False, attention_type="soft",
        attention_targets=False,
        n_graphs=5):
    low_res_embedder = TripleMNISTLowResEmbedder()
    attention_box = TripleMNISTSoftAttentionBox(attention_layer_type,
                                                attention_type=attention_type)
    core_proposal_layer = TripleMNISTCoreAndProposalLayer()

    net = FullNet(attention_box=attention_box,
                  low_res_embedder=low_res_embedder,
                  core_proposal_layer=core_proposal_layer)
    if cuda:
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=step_size)
    lossCriterion = nn.NLLLoss()
    attentionLossCriterion = nn.NLLLoss()
    dataGenerator = TripleMNISTImagesGenerator(batch_size=batch_size,
                                               noise_SD=noise_SD,
                                               single_target=True,
                                               cuda=cuda,
                                               n_locations=13,
                                               give_locations=True)
    lossTracker = trackStat("Loss")

    validation_batch = next(dataGenerator)

    net.train()
    for i in range(1, iterations+1):
        optimizer.zero_grad()

        images, targets, true_locations = next(dataGenerator)

        proposed = net(images)

        attention_locations = net.getAttentionSummary()
        ideal_attention_locations = give_ideal_locations(true_locations,
                                                         n_locations=13)

        actual_loss = lossCriterion(proposed, targets)
        if attention_targets:
            attention_loss = attentionLossCriterion(attention_locations, ideal_attention_locations)
            loss = actual_loss + attention_loss
        else:
            loss = actual_loss
        loss.backward()

        optimizer.step()

        lossTracker.add(actual_loss, i)

        print("{} iterations: Loss is {}\n".format(i, actual_loss.data[0]))

        if i in map(lambda x: int(iterations*x),
                    np.arange(0, 1.01, 1/n_graphs)):
            graphic = analysePredictions(net, validation_batch)
            if graphics_path is None:
                graphic.show()
            else:
                graphic.save(graphics_path +
                             "/predictions_after_{}.png".format(str(i).zfill(7)),
                             "PNG")

    images, targets, _ = next(dataGenerator)

    if graphics_path:
        lossTracker.plot(save_file=graphics_path +
                         "/loss_plot_for_{}.png".format(iterations))
    else:
        lossTracker.plot()
