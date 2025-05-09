import math
import time

import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


    # def fasterprune(
    #     self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, mask=None
    # ):
    #     W = self.layer.weight.data.clone()
    #     if isinstance(self.layer, nn.Conv2d):
    #         W = W.flatten(1)
    #     if isinstance(self.layer, transformers.Conv1D):
    #         W = W.t()
    #     W = W.float()

    #     tick = time.time()

    #     H = self.H
    #     del self.H
    #     dead = torch.diag(H) == 0
    #     H[dead, dead] = 1
    #     W[:, dead] = 0

    #     Losses = torch.zeros(self.rows, device=self.dev)

    #     try:
    #         H_tmp = H.clone()
    #         damp = percdamp * torch.mean(torch.diag(H_tmp))
    #         diag = torch.arange(self.columns, device=self.dev)
    #         H_tmp[diag, diag] += damp
    #         H_tmp = torch.linalg.cholesky(H_tmp)
    #         H_tmp = torch.cholesky_inverse(H_tmp)
    #         H_tmp = torch.linalg.cholesky(H_tmp, upper=True)
    #         Hinv = H_tmp
            
            
    #     except torch._C._LinAlgError:
    #         print("The matrix is not postive-definite, try a larger percdamp!")
    #         percdamp = 0.1
    #         damp = percdamp * torch.mean(torch.diag(H))
    #         diag = torch.arange(self.columns, device=self.dev)
    #         H[diag, diag] += damp
    #         H = torch.linalg.cholesky(H)
    #         H = torch.cholesky_inverse(H)
    #         H = torch.linalg.cholesky(H, upper=True)
    #         Hinv = H

    #     # mask = None

    #     for i1 in range(0, self.columns, blocksize):
    #         i2 = min(i1 + blocksize, self.columns)
    #         count = i2 - i1

    #         W1 = W[:, i1:i2].clone()
    #         Q1 = torch.zeros_like(W1)
    #         Err1 = torch.zeros_like(W1)
    #         Losses1 = torch.zeros_like(W1)
    #         Hinv1 = Hinv[i1:i2, i1:i2]

    #         if prune_n == 0: 
    #             if mask is not None:
    #                 mask1 = mask[:, i1:i2]
    #             else:
    #                 tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
    #                 thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
    #                 mask1 = tmp <= thresh
    #         else:
    #             mask1 = torch.zeros_like(W1) == 1

    #         for i in range(count):
    #             w = W1[:, i]
    #             d = Hinv1[i, i]

    #             if prune_n != 0 and i % prune_m == 0:
    #                 tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
    #                 mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

    #             q = w.clone()
    #             q[mask1[:, i]] = 0

    #             Q1[:, i] = q
    #             Losses1[:, i] = (w - q) ** 2 / d ** 2

    #             err1 = (w - q) / d 
    #             W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    #             Err1[:, i] = err1

    #         W[:, i1:i2] = Q1
    #         Losses += torch.sum(Losses1, 1) / 2

    #         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    #     torch.cuda.synchronize()
    #     if isinstance(self.layer, transformers.Conv1D):
    #         W = W.t()
    #     self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    #modified faster_prune : added damping factor to matrix
    def fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01, mask=None):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
    
        tick = time.time()
    
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
    
        Losses = torch.zeros(self.rows, device=self.dev)
    
        # Ensure the matrix is positive-definite by adding a damping factor
        min_diag = torch.min(torch.diag(H))
        if min_diag <= 0:
            print(f"Adjusting diagonal elements: min_diag={min_diag}")
            H += torch.eye(self.columns, device=self.dev) * (abs(min_diag) + 1e-5)
    
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
    
        try:
            H_tmp = H.clone()
            H_tmp = torch.linalg.cholesky(H_tmp)
        except torch._C._LinAlgError:
            print("Increasing damping factor due to Cholesky failure...")
            for factor in [0.1, 0.2, 0.5, 1.0]:  # Try increasing damping step by step
                damp = factor * torch.mean(torch.diag(H))
                H[diag, diag] += damp
                try:
                    H_tmp = H.clone()
                    H_tmp = torch.linalg.cholesky(H_tmp)
                    break  # Exit loop if successful
                except torch._C._LinAlgError:
                    continue  # Try next damping factor
            else:
                raise ValueError("Cholesky decomposition failed even with high damping!")
    
        H_tmp = torch.cholesky_inverse(H_tmp)
        H_tmp = torch.linalg.cholesky(H_tmp, upper=True)
        Hinv = H_tmp
    
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
    
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
    
            if prune_n == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1
    
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
    
                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
    
                q = w.clone()
                q[mask1[:, i]] = 0
    
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
    
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
    
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
    
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    
        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        self.H = None
        torch.cuda.empty_cache()