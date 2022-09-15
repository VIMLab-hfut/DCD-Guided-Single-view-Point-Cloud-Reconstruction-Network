import argparse
import torch
import os
import time
import imageio
import numpy as np
import torchvision.transforms as tfs
import sklearn.cluster as cls

from model.model import Model
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from collections import defaultdict

from datasets.dataset_load import ShapeNet55
from datasets.mv_dataset import ShapeNetPoint
from metrics.evaluation_metrics import distChamferCUDA
from metrics.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D



_transform = tfs.Compose([
    tfs.CenterCrop(550),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])

_transform_shape = tfs.Compose([
    tfs.CenterCrop(256),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])
def sphere_noise(batch, num_pts, device):
    with torch.no_grad():
        theta = 2 * np.pi * torch.rand(batch, num_pts, device=device)
        phi = torch.acos(1 - 2 * torch.rand(batch, num_pts, device=device))
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)
def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_dcd(x, gt, alpha=40, n_lambda=0.1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res
def main(conf):

    conf.name = 'shapenet'
    checkpoint_path = os.path.join(conf.model_path, conf.name)
    checkpoint_imgs = os.path.join(checkpoint_path, 'images')
    train_log = os.path.join(checkpoint_path, 'train.log')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(checkpoint_imgs):
        os.mkdir(checkpoint_imgs)

    root, ply_root, tgt_category = conf.root, conf.proot, conf.cat
    tgt_category = tgt_category

    if conf.dataset == 'shapenet':
        mv_ds = ShapeNet55(root, 'train', transform=_transform_shape)
        mv_ds_test = ShapeNet55(root, 'test', transform=_transform_shape, number_of_view=12)
    else:
        raise RuntimeError(f'Dataset is suppose to be [modelnet|shapenet], but {conf.dataset} is given')

    ds_loader = DataLoader(mv_ds, batch_size=conf.batch_size, drop_last=True, shuffle=True)
    ds_loader_test = DataLoader(mv_ds_test, batch_size=conf.batch_size)
    num_classes = len(mv_ds.classes)

    print(f'Dataset summary : Categories: {mv_ds.classes} with length {len(mv_ds)}')
    print(f'Num of classes is {len(mv_ds.classes)}')
    with open(train_log, 'a+') as f:
        f.write('Dataset summary : Categories: {} with length {}\n'.format(mv_ds.classes, len(mv_ds)) )
        f.write('Num of classes is {}\n'.format(len(mv_ds.classes)))


    # Initialize Model
    model = Model(conf)




    print('Start Training 2D to 3D -------------------------------------------')
    with open(train_log, 'a+') as f:
        f.write('Start Training 2D to 3D -------------------------------------------\n')

    optimizer = Adam(
        model.parameters(),
        lr=conf.lrate,
        betas=(.9, .999),
        weight_decay=0.00001
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=int(conf.nepoch / 3), gamma=.5)

    start_time = time.time()

    for i in range(conf.nepoch):
        total_loss = 0.
        print('Start Epoch {}'.format(str(i + 1)))
        with open(train_log, 'a+') as f:
            f.write('Start Epoch {}\n'.format(str(i + 1)))


        for idx, (multi_view, pc, _) in enumerate(ds_loader):
            # Get input image and add gaussian noise
            mv = np.stack(multi_view, axis=1).squeeze(axis=1)
            #mv = np.stack(multi_view)
            mv = torch.from_numpy(mv).float()

            mv = Variable(mv.cuda())
            pc = Variable(pc.cuda())  # BatchSize * 2048, currently

            # Optimize process
            optimizer.zero_grad()
            noise = sphere_noise(conf.batch_size, num_pts=2048, device=conf.device)

            syn_pc = model(mv,noise)

            dcd = calc_dcd(syn_pc, pc)
            dcd_loss = dcd[0]

            loss = dcd_loss.sum()

            total_loss += loss.detach().item()

            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    'Epoch %d Batch [%2d/%2d] Time [%3.2fs] Recon Nat %.10f' %
                    (i + 1, idx + 1, len(ds_loader), duration, loss.item() / float(conf.batch_size)))
                with open(train_log, 'a+') as f:
                    f.write('Epoch %d Batch [%2d/%2d] Time [%3.2fs] Recon Nat %.10f\n' %
                    (i + 1, idx + 1, len(ds_loader), duration, loss.item() / float(conf.batch_size)))

        print('Epoch {}  -- Recon Nat {}'.format(str(i + 1), total_loss / float(len(mv_ds))))
        with open(train_log, 'a+') as f:
            f.write('Epoch {}  -- Recon Nat {}\n'.format(str(i + 1), total_loss / float(len(mv_ds))))

        # Save model configuration
        if conf.save_interval > 0 and i % opt.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_path,
                                                        '{0}_iter_{1}.pt'.format(conf.name, str(i + 1))))

        # Validate the model on test split
        if conf.sample_interval > 0 and i % opt.sample_interval == 0:
            with torch.no_grad():
                cd, emd = shapenet_validate(model, ds_loader_test)
                print("Chamfer Distance  :%s" % cd)
                print("Earth Mover Distance :%s" % emd)
                with open(train_log, 'a+') as f:
                    f.write("Chamfer Distance  :%s\n" % cd)
                    f.write("Earth Mover Distance :%s\n" % emd)

            model.train()

        scheduler.step()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=101, help='number of epochs to train for')
    parser.add_argument('--random_seed', action="store_true", help='Fix random seed or not')
    parser.add_argument('--lrate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=120, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=140, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=145, help='learning rate decay 2')
    parser.add_argument('--device', type=str, default='cuda', help='Gpu usage')
    parser.add_argument('--dim_template', type=int, default=2, help='Template dimension')

    # Data
    parser.add_argument('--number_points', type=int, default=2048,
                        help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument('--prototypes_npy', type=str, default='NOF', help='Path of the prototype npy file')

    # Save dirs and reload
    parser.add_argument('--name', type=str, default="0", help='training name')
    parser.add_argument('--dir_name', type=str, default="", help='name of the log folder.')

    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--nb_primitives', type=int, default=16, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SQUARE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument('--number_points_eval', type=int, default=2048,
                        help='Number of points generated by atlasnet (rounded to the nearest squared number) ')
    parser.add_argument("--remove_all_batchNorms", action="store_true", help="Replace all batchnorms by identity")

    parser.add_argument('--bottleneck_size', type=int, default=512, help='dim_out_patch')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='dim_out_patch')


    # Loss
    parser.add_argument('--no_metro', action="store_true", help='Compute metro distance')

    # Additional arguments
    parser.add_argument('--root', type=str, required=True, help='The path of multi-view dataset')
    parser.add_argument('--proot', type=str, required=True, help='The path of corresponding pc dataset')
    parser.add_argument('--cat', type=str, required=True, help='Target category')
    parser.add_argument('--model_path', type=str, default='../results/result')

    parser.add_argument('--sample_interval', type=int, default=10, help='The gap between each sampling process')
    parser.add_argument('--save_interval', type=int, default=10, help='The gap between each model saving')

    parser.add_argument('--from_scratch', action="store_true", help='Train the point_feature_extractor from scratch')
    parser.add_argument('--reclustering', action="store_true", help='Flag that controls the re-clustering behavior')
    parser.add_argument('--dataset', type=str, default='shapenet', help='The dataset to use, chose from [modelnet|shapenet]')

    opt = parser.parse_args()
    main(opt)
