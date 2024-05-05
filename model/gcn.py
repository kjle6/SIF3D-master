import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        support = torch.matmul(x, self.weight)
        # print(self.att.data.shape)
        # print(self.att.data.abs().mean(dim=1))
        y = torch.matmul(self.att, support)
        if self.bias is not None:
            return y + self.bias
        else:
            return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout=0.1, bias=True, node_n=48):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, config, input_feature=16, num_stage=9, node_n=69):

        super(GCN, self).__init__()
        self.in_frames = config.input_seq_len
        self.out_frames = config.output_seq_len
        self.num_stage = num_stage
        self.node = node_n

        self.dct = DCT_Transform(config.input_seq_len + config.output_seq_len)
        self.idct = IDCT_Transform(config.input_seq_len + config.output_seq_len)

        self.gc1 = GraphConvolution(input_feature, config.motion_hidden_dim, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * config.motion_hidden_dim)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(config.motion_hidden_dim, p_dropout=config.dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(config.motion_hidden_dim, input_feature, node_n=node_n)
        self.do = nn.Dropout(config.dropout)
        self.act_f = nn.Tanh()

        self.gc7.weight.data *= 0.1
        self.gc7.att.data *= 0.1
        self.gc7.bias.data *= 0.1

    def forward(self, x):
        x = self.dct(x, dct_used=self.in_frames+self.out_frames)
        y = self.gc1(x)
        bs, node, C = y.shape
        y = self.bn1(y.view(bs, -1)).view(bs, node, C)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)
        y = self.gc7(y)
        y = y[:, :self.node] + x
        y = self.idct(y, self.node // 3, 3, dct_used=self.in_frames+self.out_frames)
        return y


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class DCT_Transform(nn.Module):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        dct_m, _ = get_dct_matrix(N=frame)
        self.dct_m = torch.from_numpy(dct_m).float()

    def forward(self, x, dct_used=16):
        self.dct_m = self.dct_m.to(x.device)
        batch, frame, node, dim = x.data.shape

        x = x.transpose(0, 1).reshape(frame, -1).contiguous()
        if frame < self.frame:
            # assert False
            x = torch.cat([x, x[frame-1:].repeat(self.frame - frame, 1)], dim=0)
        x_dct_seq = torch.matmul(self.dct_m[0:dct_used, :], x)
        x_dct_seq = x_dct_seq.transpose(0, 1).reshape(batch, node * dim, dct_used).contiguous()

        return x_dct_seq


class IDCT_Transform(nn.Module):
    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        _, idct_m = get_dct_matrix(N=frame)
        self.idct_m = torch.from_numpy(idct_m).float()

    def forward(self, x, node, dim, dct_used=16):
        self.idct_m = self.idct_m.to(x.device)

        x = x.view(-1, dct_used).transpose(1, 0)
        pred_3d = torch.matmul(self.idct_m[:, 0:dct_used], x)
        pred_3d = pred_3d.reshape(self.frame, -1, node, dim).transpose(0, 1).contiguous()
        return pred_3d
