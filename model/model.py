import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import neuralnet_pytorch as nnt
import torch.nn.functional as F
from model.img_encoder import vgg16_bn
from layers.multihead_ct_adain import forward_style
from layers.utils import AdaIn1dUpd
from layers.curve_util import CIC

class Model(nn.Module):

    def __init__(self, configuration, num_pts=2048, adain=True):
        super(Model, self).__init__()
        self.opt = configuration
        self.num_points = num_pts
        self.batch_size = self.opt.batch_size
        self.model_dim = 512
        self.num_latent = 512
        self.device = self.opt.device

        self.img_feature_extractor = vgg16_bn()

        self.img_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.up_sampler = nn.Upsample(scale_factor=4)

        self.activation = nn.ReLU(inplace=True)
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=self.model_dim, kernel_size=1, bias=False)


        self.adain_512 = AdaIn1dUpd(self.model_dim, num_latent=self.num_latent)

        self.start = nn.Sequential(nn.Conv1d(in_channels=3,
                                             out_channels=self.model_dim, kernel_size=1, bias=False),
                                   AdaIn1dUpd(self.model_dim, num_latent=self.num_latent),
                                   nn.ReLU(True))

        self.conv1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, bias=False)


        self.cic1 = CIC(npoint=2048, radius=0.05, k=20, in_channels=64, output_channels=128, bottleneck_ratio=2,
                        mlp_num=1, curve_config=[100, 5])
        self.cic = CIC(npoint=2048, radius=0.05, k=20, in_channels=128, output_channels=256, bottleneck_ratio=2,
                       mlp_num=1, curve_config=[100, 5])

        self.final = nn.Sequential(nn.Conv1d(in_channels=256,
                                             out_channels=128, kernel_size=1, bias=False),

                                   nn.Conv1d(in_channels=128,
                                             out_channels=3, kernel_size=1),
                                   nn.Tanh())


    def forward(self, x, noise, img_flag=True):
        if img_flag:
            # _, _, _, _, x5 = self.img_feature_extractor(x);
            x5 = self.img_feature_extractor(x)
            x5 = self.img_pool(x5).squeeze(-1).squeeze(-1)


            noise_feat = self.conv1d(noise)  # 初始点云的特征
            x5 = self.adain_512(noise_feat, x5)  # 风格迁移特征
            x5 = self.activation(x5)


            x = self.activation(self.conv1(x5))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))

            _, x = self.cic1(noise, x)
            # a = x
            _, x = self.cic(noise, x)

            x = forward_style(self.final, x, x5)

            return x.transpose(1, 2).contiguous()


    def _project(self, img_feats, xs, ys):
        x, y = xs.flatten(), ys.flatten()
        idb = torch.arange(img_feats.shape[0], device=img_feats.device)
        idb = idb[None].repeat(xs.shape[1], 1).t().flatten().long()

        x1, y1 = torch.floor(x), torch.floor(y)
        x2, y2 = torch.ceil(x), torch.ceil(y)
        q11 = img_feats[idb, :, x1.long(), y1.long()].to(img_feats.device)
        q12 = img_feats[idb, :, x1.long(), y2.long()].to(img_feats.device)
        q21 = img_feats[idb, :, x2.long(), y1.long()].to(img_feats.device)
        q22 = img_feats[idb, :, x2.long(), y2.long()].to(img_feats.device)

        weights = ((x2 - x) * (y2 - y)).unsqueeze(1)
        q11 *= weights

        weights = ((x - x1) * (y2 - y)).unsqueeze(1)
        q21 *= weights

        weights = ((x2 - x) * (y - y1)).unsqueeze(1)
        q12 *= weights

        weights = ((x - x1) * (y - y1)).unsqueeze(1)
        q22 *= weights
        out = q11 + q12 + q21 + q22
        return out.view(img_feats.shape[0], -1, img_feats.shape[1])

    def get_projection(self, img_feat, pc):
        _, _, h_, w_ = tuple(img_feat.shape)
        X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
        h = 248. * Y / Z + 111.5
        w = 248. * -X / Z + 111.5
        h = torch.clamp(h, 0., 223.)
        w = torch.clamp(w, 0., 223.)

        # x = (h / (223. / (h_ - 1.))).requires_grad_(False)
        # y = (w / (223. / (w_ - 1.))).requires_grad_(False)
        x = (h / (223. / (h_ - 1.)))
        y = (w / (223. / (w_ - 1.)))
        # x.requires_grad=False
        # y.requires_grad=False
        # print(x.is_leaf)  False

        feats = self._project(img_feat, x, y)
        return feats

    def transform(self, pc_feat, img_feat):
        pc_feat = (pc_feat - torch.mean(pc_feat, -1, keepdim=True)) / torch.sqrt(
            torch.var(pc_feat, -1, keepdim=True) + 1e-8)
        mean, var = torch.mean(img_feat, (2, 3)), torch.var(torch.flatten(img_feat, 2), 2)

        output = (pc_feat * torch.sqrt(nnt.utils.dimshuffle(var, (0, 1, 'x')) + 1e-8)) + nnt.utils.dimshuffle(mean, (
            0, 1, 'x'))
        return output

    def extract_prototypes(self):
        return np.concatenate([self.decoder[idx].extract_prototype() for idx in range(self.num_prototypes)])
