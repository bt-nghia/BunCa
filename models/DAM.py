import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from gene_ii_co_oc import load_sp_mat
from models.AsymModule import AsymMatrix


def cal_bpr_loss(pred, alpha=0.2):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    # normal bpr loss
    loss = - torch.log(torch.sigmoid(pos - negs))  # [bs]
    loss = torch.mean(loss)
    return loss


class DAM(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.sdense = nn.Linear(self.embedding_size * 2, self.embedding_size * 2)
        self.dense = nn.Linear(self.embedding_size * 2, self.embedding_size * 2)
        self.ipred = nn.Linear(self.embedding_size * 2, 1)
        self.bpred = nn.Linear(self.embedding_size * 2, 1)

    def propagate(self, test=False):
        pass

    def forward(self, x):
        pass


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom,
                                                                                   bundles_feature_non_atom.t())
        return scores