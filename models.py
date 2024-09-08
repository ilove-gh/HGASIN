import math
import os
import torch.nn.init as init
import numpy as np
import torch
import yaml

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from GDCModel import HeatDataset, PPRDataset


class HGDGC(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, is_enhanced=True, method='PPR',
                 device='cuda',t = 5,k = 128,eps=0.0001, alpha_h = 0.08):
        super(HGDGC, self).__init__()
        self.model_R = GCNII(nfeat=nfeat, nlayers=nlayers, nhidden=nhidden, dropout=dropout, lamda=lamda, alpha=alpha,
                             variant=variant)
        self.model_L = GCNII(nfeat=nfeat, nlayers=nlayers, nhidden=nhidden, dropout=dropout, lamda=lamda, alpha=alpha,
                             variant=variant)
        self.model_UCN_1 = GCN(nfeat=nfeat, nlayers=1, nhidden=nhidden, nclass=1, dropout=dropout)
        self.model_UCN_2 = GCN(nfeat=nfeat, nlayers=1, nhidden=nhidden, nclass=1, dropout=dropout)
        self.classify = nn.Linear(nhidden, nclass)
        self.is_enhanced = is_enhanced
        self.device = device
        self.attention = Attention(nhidden)
        self.heat_t = t
        self.heat_eps = eps
        self.ppr_k = k
        self.ppr_alpha_h = alpha_h
        if is_enhanced:
            with open(os.getcwd() + '/' + 'config.yaml', 'r') as c:
                config = yaml.safe_load(c)

            Heat_difussion = HeatDataset(
                t=config['heat']['t'],
                k=config['heat']['k'],
                eps=config['heat']['eps']
            )

            PPR_difussion = PPRDataset(
                alpha=self.ppr_alpha_h,
                k= self.ppr_k
            )
            if method == 'PPR':
                self.enhanced_model = PPR_difussion
            elif method == 'Heat':
                self.enhanced_model = Heat_difussion
            else:
                print(
                    'The value of the metsod parameter is {}, and it is stronger by default in the PPR mode.'.format(
                        method))
                self.enhanced_model = PPR_difussion

    def forward(self, features, adj):
        R_adj, R_features, L_adj, L_features, ucn_features, ucn_adj, cross_ucn_adj, no_cross_ucn_adj, ucn_row_set_list = self.divide_brain_network(
            adj,
            features)
        # 使用热核增强邻接矩阵
        if self.is_enhanced:
            R_adj = self.diffusion_enhancement(R_adj)
            L_adj = self.diffusion_enhancement(L_adj)

        R_adj = torch.from_numpy(R_adj).to(self.device)
        R_features = torch.from_numpy(R_features).to(self.device)
        L_adj = torch.from_numpy(L_adj).to(self.device)
        L_features = torch.from_numpy(L_features).to(self.device)
        ucn_features = torch.from_numpy(ucn_features).to(self.device)

        # ucn_adj = torch.from_numpy(ucn_adj).to(self.device)
        cross_ucn_adj = torch.from_numpy(cross_ucn_adj).to(self.device)
        no_cross_ucn_adj = torch.from_numpy(no_cross_ucn_adj).to(self.device)

        X_R = self.model_R(R_features, R_adj)
        X_L = self.model_L(L_features, L_adj)

        X_U_1 = self.model_UCN_1(ucn_features, cross_ucn_adj)
        X_U_2 = self.model_UCN_2(ucn_features, no_cross_ucn_adj)
        X_U = (X_U_1 + X_U_2) / 2
        L_U_R = torch.zeros((X_R.shape[0], X_R.shape[1] * 2, X_R.shape[2])).to(self.device)
        for idx, rows_to_select in enumerate(ucn_row_set_list):
            L_U_R[idx, 0::2, :] = X_L[idx]
            L_U_R[idx, 1::2, :] = X_R[idx]
            X_LR_selected_rows = L_U_R[idx, rows_to_select, :]
            X_U_selected_rows = X_U[idx, rows_to_select, :]
            emb = torch.stack([X_LR_selected_rows, X_U_selected_rows], dim=1)
            emb, att = self.attention(emb)
            L_U_R[idx, rows_to_select, :] = emb
        # X_contact = torch.cat((X_R,X_L,X_U),dim=1)
        hg = L_U_R.sum(dim=1, dtype=torch.float)
        return self.classify(hg)

    def divide_brain_network(self, adj, features):
        R_adj = adj[:, ::2, ::2].copy()
        R_features = features[:, ::2, :].copy()
        L_adj = adj[:, 1::2, 1::2].copy()
        L_features = features[:, 1::2, :].copy()

        ucn_row_set_list = []
        ucn_adj_list = []
        cross_ucn_adj_list = []
        no_cross_ucn_adj_list = []
        features_list = []
        for idx in range(adj.shape[0]):
            ucn_adj, cross_ucn_adj, no_cross_ucn_adj, ucn_row_set = self.compute_ucn_network_neighbor(adj[idx])
            ucn_row_set_list.append(ucn_row_set)
            ucn_adj_list.append(ucn_adj)
            cross_ucn_adj_list.append(cross_ucn_adj)
            no_cross_ucn_adj_list.append(no_cross_ucn_adj)
            features_list.append(self.mask_matrix(features[idx], ucn_row_set))
        ucn_features = np.stack(features_list, axis=0)
        ucn_adj = np.stack(ucn_adj_list, axis=0)
        cross_ucn_adj = np.stack(cross_ucn_adj_list, axis=0)
        no_cross_ucn_adj = np.stack(no_cross_ucn_adj_list, axis=0)

        return R_adj, R_features, L_adj, L_features, ucn_features, ucn_adj, cross_ucn_adj, no_cross_ucn_adj, ucn_row_set_list

    def diffusion_enhancement(self, adj):
        for idx in range(adj.shape[0]):
            adj[idx] = self.enhanced_model.process(adj[idx])
        return adj

    def compute_ucn_network_neighbor(self, matrix):
        """
        只保留交叉节点以及交叉节点相邻的节点
        :param matrix:
        :return:
        """
        ucn_row_set = set()
        # Traverse the upper triangular part of the matrix
        for rol in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if matrix[rol, col] != 0 and (not self.is_parity_same(rol, col)):
                    ucn_row_set.add(rol)
        lst = range(matrix.shape[0])
        # 使用列表推导来过滤掉 list1 中存在于 list2 的元素
        result = [item for item in lst if item not in list(ucn_row_set)]
        matrix[list(result), :] = 0
        matrix[:, list(result)] = 0
        cross_matrix = matrix.copy()
        no_cross_matrix = matrix.copy()
        for rol in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if matrix[rol, col] != 0 and (not self.is_parity_same(rol, col)):
                    no_cross_matrix[rol, col] = 0

        for rol in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                if matrix[rol, col] != 0 and self.is_parity_same(rol, col):
                    cross_matrix[rol, col] = 0

        ucn_row_set = sorted(ucn_row_set)
        return matrix, cross_matrix, no_cross_matrix, ucn_row_set


    def is_parity_same(self, num1, num2):
        '''判断奇偶性是否相同'''
        if num1 % 2 == num2 % 2:
            return True
        else:
            return False

    def mask_matrix(self, matrix, mask_row_list: list):
        """
        掩码矩阵，保留矩阵mask_row_list中的行，其他行置为0
        :param matrix:
        :param mask_row_list:
        :return:
        """
        mask = np.ones(matrix.shape[0], dtype=bool)
        mask[mask_row_list] = False
        matrix[mask, :] = 0
        return matrix

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class GCN(nn.Module):
    # 版本3，可设置多层卷积。
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhidden)
        self.gc2 = GraphConvolution(nhidden, nhidden)
        self.dropout = dropout
        # 第一层卷积
        self.classify = torch.nn.Sequential(
            nn.Linear(nclass, nclass)
        )
        self.dropout = dropout

    def forward(self, X, adj):
        X = F.relu(self.gc1(X, adj))
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.gc2(X, adj)
        X = F.relu(X)
        return F.softmax(X, dim=2)

    def l2_loss(self):
        """
        只对第一层卷积的参数计算范数
        :return:
        """
        layer = self.convs.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 原始卷积模块
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution2(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        # self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        # 随机删除部分神经元
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        # layer_inner = layer_inner.mean(dim=1, dtype=torch.float)
        # classify = self.fcs[-1](layer_inner)
        return F.softmax(layer_inner, dim=2)


class GraphConvolution2(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution2, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.matmul(adj, input)
        # is variant的计算，进行了简化，可以理解为加速计算，与论文公式略有不同。
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.matmul(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output