import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .images import TripleMNISTImagesGenerator
from .nn import TripleMNISTLowResEmbedder,\
               TripleMNISTAttentionBox,\
               TripleMNISTCoreAndProposalLayer
from ..base import FullNet
from .analyse import analysePredictions, trackStat


def run(step_size=1e-3,
        batch_size=10,
        noise_SD=20,
        iterations=10,
        graphics_path=None,
        cuda=False):
    low_res_embedder = TripleMNISTLowResEmbedder()
    attention_box = TripleMNISTAttentionBox()
    core_proposal_layer = TripleMNISTCoreAndProposalLayer()

    net = FullNet(attention_box=attention_box,
                  low_res_embedder=low_res_embedder,
                  core_proposal_layer=core_proposal_layer)
    if cuda:
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=step_size)
    lossCriterion = nn.NLLLoss()
    dataGenerator = TripleMNISTImagesGenerator(batch_size=batch_size,
                                               noise_SD=noise_SD,
                                               single_target=True,
                                               cuda=cuda)
    lossTracker = trackStat("Loss")

    validation_batch = next(dataGenerator)

    net.train()
    for i in range(1, iterations+1):
        optimizer.zero_grad()

        images, targets = next(dataGenerator)

        proposed = net(images)

        loss = lossCriterion(proposed, targets)
        loss.backward()
        print(loss)

        optimizer.step()

        lossTracker.add(loss, i)

        print("Loss is {}\n".format(loss.data[0]))

        if i in map(lambda x: int(iterations*x),
                    [0.1, 0.2, 0.4, 0.6, 0.8, 1]):
            graphic = analysePredictions(net, validation_batch)
            if graphics_path is None:
                graphic.show()
            else:
                graphic.save(graphics_path +
                             "/predictions_after_{}.png".format(i),
                             "PNG")

    images, targets = next(dataGenerator)
    print(net(images), targets)
    print(net.getAttentionSummary())

    if graphics_path:
        lossTracker.plot(save_file=graphics_path +
                         "/loss_plot_for_{}.png".format(iterations))
    else:
        lossTracker.plot()
