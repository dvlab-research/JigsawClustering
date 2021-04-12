# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.distributed as dist
import diffdist

from .losses import SupCluLoss


class JigClu(nn.Module):
    def __init__(self, base_encoder, dim=128, T=0.07):
        """
        dim: feature dimension (default: 128)
        T: softmax temperature (default: 0.07)
        """
        super(JigClu, self).__init__()

        self.criterion_clu = SupCluLoss(temperature=T)

        self.criterion_loc = nn.CrossEntropyLoss()

        # num_classes is the output fc dimension
        self.encoder = base_encoder(num_classes=dim)

        dim_mlp = self.encoder.fc_clu.weight.shape[1]
        self.encoder.fc_clu = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc_clu)

    @torch.no_grad()
    def _batch_gather_ddp(self, images):
        """
        gather images from different gpus and shuffle between them
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        images_gather = []
        for i in range(4):
            batch_size_this = images[i].shape[0]
            images_gather.append(concat_all_gather(images[i]))
            batch_size_all = images_gather[i].shape[0]
        num_gpus = batch_size_all // batch_size_this


        n,c,h,w = images_gather[0].shape
        permute = torch.randperm(n*4).cuda()
        torch.distributed.broadcast(permute, src=0)
        images_gather = torch.cat(images_gather, dim=0)
        images_gather = images_gather[permute,:,:,:]
        col1 = torch.cat([images_gather[0:n], images_gather[n:2*n]], dim=3)
        col2 = torch.cat([images_gather[2*n:3*n], images_gather[3*n:]], dim=3)
        images_gather = torch.cat([col1, col2], dim=2)
 

        bs = images_gather.shape[0] // num_gpus
        gpu_idx = torch.distributed.get_rank()

        return images_gather[bs*gpu_idx:bs*(gpu_idx+1)], permute, n

    def forward(self, images, progress):
        images_gather, permute, bs_all = self._batch_gather_ddp(images)


        # compute features
        q = self.encoder(images_gather) 

        q_gather = concat_all_gather(q)
        n,c,h,w = q_gather.shape
        c1,c2 = q_gather.split([1,1],dim=2)
        f1,f2 = c1.split([1,1],dim=3)
        f3,f4 = c2.split([1,1],dim=3)
        q_gather = torch.cat([f1,f2,f3,f4],dim=0)
        q_gather = q_gather.view(n*4,-1)

        # clustering branch
        label_clu = permute % bs_all
        q_clu = self.encoder.fc_clu(q_gather)
        q_clu = nn.functional.normalize(q_clu, dim=1)
        loss_clu = self.criterion_clu(q_clu, label_clu)

        # location branch
        label_loc = torch.LongTensor([0]*bs_all+[1]*bs_all+[2]*bs_all+[3]*bs_all).cuda()
        label_loc = label_loc[permute]
        q_loc = self.encoder.fc_loc(q_gather)
        loss_loc = self.criterion_loc(q_loc, label_loc)

        return loss_clu, loss_loc

def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    tensors_gather = diffdist.functional.all_gather(tensors_gather, tensor, next_backprop=None, inplace=True)

    output = torch.cat(tensors_gather, dim=0)
    return output
