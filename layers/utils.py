import torch
from torch import nn, einsum

import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# from pytorch3d.transforms.so3 import so3_exponential_map
def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample + pad]
    return idx.int()


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class VolTransformer(nn.Module):
    def __init__(self, heads, scales=False):
        super().__init__()
        self.heads = heads

        self.log_R = nn.Parameter(torch.randn(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)

        self.do_scales = scales

        if self.do_scales:
            self.scales = nn.Parameter(torch.ones(self.heads, 3, dtype=torch.float32),
                                       requires_grad=True)

    def forward(self, pcd):
        # pcd [b, h, c, p]
        pcd = pcd + self.shift[None, :, :, None]

        pcd = torch.einsum('bhcp,hcn->bhnp', [pcd, so3_exponential_map(self.log_R)])

        if self.do_scales:
            return pcd * self.scales[None, :, :, None]
        else:
            return pcd


class PlaneTransformer(nn.Module):
    def __init__(self, heads, scales=False):
        super().__init__()
        self.heads = heads

        self.log_R = nn.Parameter(torch.randn(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)
        self.shift = nn.Parameter(torch.zeros(self.heads, 3, dtype=torch.float32),
                                  requires_grad=True)

        self.do_scales = scales

        if self.do_scales:
            self.scales = nn.Parameter(torch.ones(self.heads, 2, dtype=torch.float32),
                                       requires_grad=True)

    def forward(self, pcd):
        # pcd [b, h, c, p]
        pcd = pcd + self.shift[None, :, :, None]
        pcd = torch.einsum('bhcp,hcn->bhnp', [pcd, so3_exponential_map(self.log_R)])

        if self.do_scales:
            return pcd[:, :, :2] * self.scales[None, :, :, None]
        else:
            return pcd[:, :, :2]


def forward_stats(input, module, type):
    whereto = []
    current = input

    for layer in module:
        if isinstance(layer, type):
            current, lattice_size = layer(current)
            if isinstance(lattice_size, list):
                whereto += lattice_size
            else:
                whereto.append(lattice_size)
            continue

        current = layer(current)

    return current, whereto


class AdaIn1dUpd(nn.Module):
    def __init__(self, num_features, num_latent):
        super().__init__()
        self.num_features = num_features
        self.num_latent = num_latent

        self.instance_norm = nn.InstanceNorm1d(self.num_features, eps=1e-5, affine=False)
        self.linear = nn.Linear(self.num_latent, self.num_features * 2)

    def forward(self, x, z):
        x = self.instance_norm(x)

        var_bias = self.linear(z).reshape(-1, 2, self.num_features)
        # print(var_bias[:, 0][:, :, None].abs().mean(), var_bias[:, 0][:, :, None].abs().max(), flush=True)

        return x * (var_bias[:, 0][:, :, None] + 1) + var_bias[:, 1][:, :, None]


def trilinear_coords(keys):
    assert (keys.shape[1] == 3)

    spread = torch.from_numpy(np.array([[0, 0, 0],
                                        [1, 0, 0],
                                        [0, 1, 0],
                                        [1, 1, 0],
                                        [0, 0, 1],
                                        [1, 0, 1],
                                        [0, 1, 1],
                                        [1, 1, 1]])).to(keys.device)

    floored = keys.floor()
    ix, iy, iz = keys[:, 0], keys[:, 1], keys[:, 2]
    ix_tnw, iy_tnw, iz_tnw = floored[:, 0], floored[:, 1], floored[:, 2]

    ix_tne = ix_tnw + 1
    iy_tne = iy_tnw
    iz_tne = iz_tnw

    ix_tsw = ix_tnw
    iy_tsw = iy_tnw + 1
    iz_tsw = iz_tnw

    ix_tse = ix_tnw + 1
    iy_tse = iy_tnw + 1
    iz_tse = iz_tnw

    ix_bnw = ix_tnw
    iy_bnw = iy_tnw
    iz_bnw = iz_tnw + 1

    ix_bne = ix_tnw + 1
    iy_bne = iy_tnw
    iz_bne = iz_tnw + 1

    ix_bsw = ix_tnw
    iy_bsw = iy_tnw + 1
    iz_bsw = iz_tnw + 1

    ix_bse = ix_tnw + 1
    iy_bse = iy_tnw + 1
    iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    coordinates = torch.stack([tnw, tne, tsw, tse, bnw, bne, bsw, bse], dim=1)

    return coordinates, floored[:, None].long() + spread[:, :, None]


def bilinear_coords(keys):
    assert (keys.shape[1] == 2)

    spread = torch.from_numpy(np.array([[0, 0],
                                        [1, 0],
                                        [0, 1],
                                        [1, 1]])).to(keys.device)

    floored = keys.floor()
    ix, iy = keys[:, 0], keys[:, 1]
    ix_nw, iy_nw = floored[:, 0], floored[:, 1]

    ix_ne = ix_nw + 1
    iy_ne = iy_nw

    ix_sw = ix_nw
    iy_sw = iy_nw + 1

    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    coordinates = torch.stack([nw, ne, sw, se], dim=1)

    return coordinates, floored[:, None].long() + spread[:, :, None]


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        Conv = nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = input
        out = gamma * out + beta

        return out
