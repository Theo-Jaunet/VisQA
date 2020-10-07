import torch
from torch import nn

class my_triplet(nn.Module):
    """ Kind of triplet loss where input is already the embedding distance for positive and negative"""
    def __init__(self, margin):
        super(my_triplet, self).__init__()
        self.margin = margin

    def forward(self, pos_weights, neg_weights, similarity):
        # Positive example: it is a semi-hard distribution over all the visual objects
        pos_term = (pos_weights * similarity).sum(-1)
        # Negative example: pick the hardest sample among negative objects
        # hardest sample is the one with th ebiggest similarity
        neg_obj = neg_weights * similarity
        hardest_obj = torch.sort(neg_obj, descending=True, dim=-1).values[:, :, 0]
        neg_term = hardest_obj
        tloss = torch.max(input=neg_term - pos_term + self.margin, other=torch.zeros_like(pos_term))
        return tloss

    