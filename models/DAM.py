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
        self.init_emb()


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def propagate(self, y, z, task="ub"):
        x = torch.cat((y, z), dim=1)
        x = torch.relu(self.sdense(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.dense(x))
        x = F.dropout(x, p=0.5, training=self.training)

        if task == "ub":
            x = self.bpred(x)
        else:
            x = self.ipred(x)
        return x


    def forward(self, batch):
        userbs, bundles, useris, items = batch
        
        # print(bundles)
        ubs = self.users_feature[userbs].squeeze(dim=1)
        pb, nb = bundles[:, 0], bundles[:, 1]
        pb_embedding = self.bundles_feature[pb]
        nb_embedding = self.bundles_feature[nb]

        # print(ubs.shape, pb_embedding.shape, nb_embedding.shape)

        pub = self.propagate(ubs, pb_embedding, task="ub") #[bs, 1]
        nub = self.propagate(ubs, nb_embedding, task="ub")
        b_bpr_loss = torch.mean(-torch.log(torch.sigmoid(pub - nub)))


        uis = self.users_feature[useris].squeeze(dim=1)
        pi, ni = items[:, 0], items[:, 1]
        pi_embedding = self.items_feature[pi]
        ni_embedding = self.items_feature[ni]

        pui = self.propagate(uis, pi_embedding, task="ui")
        nui = self.propagate(uis, ni_embedding, task="ui")
        i_bpr_loss = torch.mean(-torch.log(torch.sigmoid(pui - nui)))

        loss = b_bpr_loss + i_bpr_loss
        return loss


    @torch.no_grad()
    def evaluate(self, users):
        # scores = []
        users_feature, bundles_feature = self.users_feature, self.bundles_feature
        # users_feature = users_feature[users]
        # for x in users_feature:
        #     sample_x = x.expand(self.num_bundles, self.embedding_size)
        #     score_each = self.propagate(sample_x, bundles_feature).reshape(1, -1)
        #     scores.append(score_each)
        # scores = torch.stack(scores, dim=1).squeeze(dim=1)
        scores = users_feature @ bundles_feature.T
        return scores