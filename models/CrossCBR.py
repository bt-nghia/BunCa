#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from gene_ii_co_oc import load_sp_mat
from models.AsymModule import AsymMatrix


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class CrossCBR(nn.Module):
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

        self.w1 = conf["w1"]
        self.w2 = conf["w2"]
        self.w3 = conf["w3"]
        self.w4 = conf["w4"]
        self.extra_layer = conf["extra_layer"]

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        self.ubi_graph = self.ub_graph @ self.bi_graph

        self.ovl_ui = self.ubi_graph.tocsr().multiply(self.ui_graph.tocsr())
        self.ovl_ui = self.ovl_ui > 0
        self.non_ovl_ui = self.ui_graph - self.ovl_ui
        # w1: 0.8, w2: 0.2
        self.ui_graph = self.ovl_ui * self.w1 + self.non_ovl_ui * self.w2

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()
        self.get_user_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()
        self.get_user_agg_graph()

        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        # light-gcn weight
        temp = self.conf["UB_coefs"]
        self.UB_coefs = torch.tensor(temp).unsqueeze(0).unsqueeze(-1).to(self.device)
        temp = self.conf["BI_coefs"]
        self.BI_coefs = torch.tensor(temp).unsqueeze(0).unsqueeze(-1).to(self.device)
        temp = self.conf["UI_coefs"]
        self.UI_coefs = torch.tensor(temp).unsqueeze(0).unsqueeze(-1).to(self.device)
        del temp
        self.a_self_loop = self.conf["self_loop"]
        self.n_head = self.conf["nhead"]
        # ii-asym matrix
        self.sw = conf["sw"]
        self.nw = conf["nw"]
        self.ibi_edge_index = torch.tensor(np.load("datasets/{}/n_neigh_ibi.npy".format(conf["dataset"]), allow_pickle=True)).to(self.device)
        self.iui_edge_index = torch.tensor(np.load("datasets/{}/n_neigh_iui.npy".format(conf["dataset"]), allow_pickle=True)).to(self.device)
        self.iui_gat_conv = Amatrix(in_dim=64, out_dim=64, n_layer=1, dropout=0.1, heads=self.n_head, concat=False, self_loop=self.a_self_loop, extra_layer=self.extra_layer)
        self.ibi_gat_conv = Amatrix(in_dim=64, out_dim=64, n_layer=1, dropout=0.1, heads=self.n_head, concat=False, self_loop=self.a_self_loop, extra_layer=self.extra_layer)

        self.iui_attn = None
        self.ibi_attn = None

    
    def construct_hyper_graph(self, threshold=10):
        ubu_graph = self.ub_graph @ self.ub_graph.T
        ubu_graph = ubu_graph > threshold

        bub_graph = self.ub_graph.T @ self.ub_graph
        bub_graph = bub_graph > threshold

        # ub_view = sp.vstack((self.ub_graph, bub_graph))
        # ub_view = sp.hstack((ub_view, sp.vstack((ubu_graph, self.ub_graph.T))))
        ub_view = sp.bmat([[self.ub_graph, bub_graph], 
                           [ubu_graph , self.ub_graph.T]])
        
        modification_ratio = self.conf["item_level_ratio"]
        
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = ub_view.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                ub_view = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.ub_hyper_propagation_graph_ori = to_tensor(laplace_transform(ub_view)).to(self.device)


    def save_asym(self):
        torch.save(self.ibi_attn, "datasets/{}/ibi_attn".format(self.conf["dataset"]))
        torch.save(self.iui_attn, "datasets/{}/iui_attn".format(self.conf["dataset"]))


    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        bi_graph = self.bi_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        bi_propagate_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bi_propagate_graph_ori = to_tensor(laplace_transform(bi_propagate_graph)).to(device)
        
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

                graph2 = bi_propagate_graph.tocoo()
                values2 = np_edge_dropout(graph2.data, modification_ratio)
                bi_propagate_graph = sp.coo_matrix((values2, (graph2.row, graph2.col)), shape=graph2.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
        self.bi_propagate_graph = to_tensor(laplace_transform(bi_propagate_graph)).to(device)


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_user_agg_graph(self):
        ui_graph = self.ui_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.ui_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            ui_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1/user_size.A.ravel()) @ ui_graph
        self.user_agg_graph = to_tensor(ui_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)

    
    def get_user_agg_graph_ori(self):
        ui_graph = self.ui_graph
        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1/user_size.A.ravel()) @ ui_graph
        self.user_agg_graph_ori = to_tensor(ui_graph).to(self.device)


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test, coefs=None):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        if coefs is not None:
            all_features = all_features * coefs
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature
    
    
    def get_IL_user_rep(self, IL_items_feature, test):
        if test:
            IL_users_feature = torch.matmul(self.user_agg_graph_ori, IL_items_feature)
        else:
            IL_users_feature = torch.matmul(self.user_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_users_feature = self.bundle_agg_dropout(IL_users_feature)

        return IL_users_feature


    def propagate(self, test=False):
        #  =============================  item level propagation  =============================
        #  ======== UI =================
        IL_items_feat, self.iui_attn = self.iui_gat_conv(self.items_feature, self.iui_edge_index, return_attention_weights=True) 
        IL_items_feat = IL_items_feat * self.nw + self.items_feature * self.sw
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, IL_items_feat, self.item_level_dropout, test, self.UI_coefs)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, IL_items_feat, self.item_level_dropout, test, self.UI_coefs)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)

        # ========== BI ================
        IL_items_feat2, self.ibi_attn = self.ibi_gat_conv(self.items_feature, self.ibi_edge_index, return_attention_weights=True) 
        IL_items_feat2 = IL_items_feat2 * self.nw + self.items_feature * self.sw
        if test:
            BIL_bundles_feature, IL_items_feature2 = self.one_propagate(self.bi_propagate_graph_ori, self.bundles_feature, IL_items_feat2, self.item_level_dropout, test, self.BI_coefs)
        else:
            BIL_bundles_feature, IL_items_feature2 = self.one_propagate(self.bi_propagate_graph, self.bundles_feature, IL_items_feat2, self.item_level_dropout, test, self.BI_coefs)
        
        # agg item -> user
        BIL_users_feature = self.get_IL_user_rep(IL_items_feature2, test)

        # w3: 0.2, w4: 0.8
        fuse_bundles_feature = IL_bundles_feature * (1 - self.w3) + BIL_bundles_feature * self.w3
        fuse_users_feature = IL_users_feature * (1 - self.w4) + BIL_users_feature * self.w4

        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.ub_hyper_propagation_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test, self.UB_coefs)
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.ub_hyper_propagation_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test, self.UB_coefs)

        users_feature = [fuse_users_feature, BL_users_feature]
        bundles_feature = [fuse_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature
    
    
    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature
        # [bs, 1+neg_num]
        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2) + torch.sum(BL_users_feature * BL_bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)

        c_losses = [u_cross_view_cl, b_cross_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()
            self.construct_hyper_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_feature, bundles_feature = self.propagate()

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores
    

class Amatrix(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=1, dropout=0.0, heads=2, concat=False, self_loop=True, extra_layer=False):
        super(Amatrix, self).__init__()
        self.num_layer = n_layer
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.self_loop = self_loop
        self.extra_layer = extra_layer
        self.convs = nn.ModuleList([AsymMatrix(in_channels=self.in_dim, 
                                              out_channels=self.out_dim, 
                                              dropout=self.dropout,
                                              heads=self.heads,
                                              concat=self.concat,
                                              add_self_loops=self.self_loop,
                                              extra_layer=self.extra_layer) 
                                              for _ in range(self.num_layer)])


    def forward(self, x, edge_index, return_attention_weights=True):
        feats = [x]
        attns = []

        for conv in self.convs:
            x, attn = conv(x, edge_index, return_attention_weights=return_attention_weights)
            feats.append(x)
            attns.append(attn)

        feat = torch.stack(feats, dim=1)
        x = torch.mean(feat, dim=1)
        return x, attns