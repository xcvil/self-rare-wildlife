# For MoCo codes
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# For the rest of the codes
# Copyright (c) ETH Zurich and EPFL ECEO lab. All Rights Reserved
import torch
import torch.nn as nn

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, two_branch=False, normlinear=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.two_branch = two_branch

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)
        self.encoder_k = base_encoder(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)

        if mlp and not two_branch:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def generate_logits_labels(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        return logits

    def forward(self, im_q, im_k, im_q2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        if self.two_branch:
            eq1 = nn.functional.normalize(q[1], dim=1) # branch 2
            q = q[0]                                   # branch 1
            # Generate logits for im_q2
            q2 = self.encoder_q(im_q2)  # queries: NxC
            eq2 = nn.functional.normalize(q2[1], dim=1) # branch 2
            q2 = nn.functional.normalize(q2[0], dim=1)  # branch 1

        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            if self.two_branch:
                ek1 = nn.functional.normalize(k[1], dim=1)
                k = k[0]
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits.
        # labels: positive key indicators
        logits = self.generate_logits_labels(q, k)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        if self.two_branch:
            logits2 = self.generate_logits_labels(q2, k)
            labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        if self.two_branch:
            return logits, labels, logits2, labels2, eq1, ek1, eq2
        return logits, labels


class LooC(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, base_head, dim=128, K=65536, m=0.999, T=0.07, mlp=False, two_branch=False, normlinear=False, shared_head=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(LooC, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.two_branch = two_branch
        self.shared_head = shared_head

        # create the encoders
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()
        # create the head
        # num_classes is the output fc dimension
        self.head_q = base_head(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)
        self.head_k = base_head(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)
        if not self.shared_head:
            self.head_q2 = base_head(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)
            self.head_k2 = base_head(num_classes=dim, two_branch=two_branch, mlp=mlp, normlinear=normlinear)

        if not mlp and not two_branch:  # hack: brute-force replacement
            dim_mlp = self.head_q.fc.weight.shape[1]
            self.head_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.head_q.fc)
            self.head_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.head_k.fc)
            if not self.shared_head:
                self.head_q2.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.head_q2.fc)
                self.head_k2.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.head_k2.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if not self.shared_head:
            for param_q, param_k in zip(self.head_q2.parameters(), self.head_k2.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        if not self.shared_head:
            for param_q, param_k in zip(self.head_q2.parameters(), self.head_k2.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def generate_logits_labels(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        return logits

    def forward(self, im_q, im_k, im_q2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)
        q = self.head_q(q)          # queries: NxC
        if self.two_branch:
            eq1 = nn.functional.normalize(q[1], dim=1) # branch 2
            q = q[0]                                   # branch 1
            # Generate logits for im_q2
            q2 = self.encoder_q(im_q2)  # queries: NxC
            if not self.shared_head:
                q2 = self.head_q2(q2)
            else:
                q2 = self.head_q(q2)
            eq2 = nn.functional.normalize(q2[1], dim=1) # branch 2
            q2 = nn.functional.normalize(q2[0], dim=1)  # branch 1
        else:
            if not self.shared_head:
                q2 = self.encoder_q(im_q2)
                q2 = self.head_q2(q2)
                q2 = nn.functional.normalize(q2, dim=1)
            else:
                q2 = self.encoder_q(im_q2)
                q2 = self.head_q(q2)
                q2 = nn.functional.normalize(q2, dim=1)

        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)
            k = self.head_k(k)      # keys: NxC
            if self.two_branch:
                ek1 = nn.functional.normalize(k[1], dim=1)
                k = k[0]
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # for k2
            # shuffle for making use of BN
            if not self.shared_head:
                im_k2, idx2_unshuffle = self._batch_shuffle_ddp(im_k)

                k2 = self.encoder_k(im_k2)
                k2 = self.head_k2(k2)  # keys: NxC
                if self.two_branch:
                    ek2 = nn.functional.normalize(k2[1], dim=1)
                    k2 = k2[0]
                k2 = nn.functional.normalize(k2, dim=1)

                # undo shuffle
                k2 = self._batch_unshuffle_ddp(k2, idx2_unshuffle)

        # compute logits.
        # labels: positive key indicators
        logits = self.generate_logits_labels(q, k)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        if not self.shared_head:
            logits2 = self.generate_logits_labels(q2, k2)
            labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()
        elif self.shared_head or self.two_branch:
            logits2 = self.generate_logits_labels(q2, k)
            labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        if not self.shared_head:
            self._dequeue_and_enqueue(k2)

        if self.two_branch:
            return logits, labels, logits2, labels2, eq1, ek1, eq2
        return logits, labels, logits2, labels2


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GroupQue(nn.Module):
    """
    Build a queue for Group Discriminative Branch
    """
    def __init__(self, dim=128, K=256*15, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(GroupQue, self).__init__()

        self.K = K
        self.T = T

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


class EvalModel(nn.Module):
    """
    Build a structured as followed:
    image -> encoder -> downstream model -> results/feature maps
    """
    def __init__(self, encoder, downstream):
        """
        encoder
        downstream
        """
        super(EvalModel, self).__init__()
        self.encoder = encoder
        self.downstream = downstream


    def forward(self, x):
        feature_vector = self.encoder(x)
        res = self.downstream(feature_vector)
        return res
