import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops import rearrange
from math import sqrt
import time

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = torch.mean(x[:,:, C//2:], dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)

def HardTopK(k, x):
    topk_results = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk_results.indices # b, k
    indices = torch.sort(indices, dim=-1).values
    return indices


class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 1000):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.k = k

    def __call__(self, x, sigma):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, sigma)



class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 1000, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise
        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        if ctx.sigma <= 1e-20:
            b, _, k, d = ctx.perturbed_output.size()
            expected_gradient = torch.zeros(b, k, d).to(grad_output.device)
        else:
            expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / (ctx.sigma)
            )

        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)

        return (grad_input,) + tuple([None] * 5)


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def extract_patches_from_indices(x, indices):
    batch_size, _, channels = x.shape
    k = indices.shape[-1]
    patches = x
    patches = batched_index_select(patches, 1, indices)
    patches = patches.contiguous().view(batch_size, k, channels)
    return patches


def extract_patches_from_indicators(x, indicators):
    indicators = rearrange(indicators, "b d k -> b k d")
    patches = torch.einsum("b k d, b d c -> b k c",
                         indicators, x)
    return patches

def min_max_norm(x):
    flatten_score_min = x.min(axis=-1, keepdim=True).values
    flatten_score_max = x.max(axis=-1, keepdim=True).values
    norm_flatten_score = (x - flatten_score_min) / (flatten_score_max - flatten_score_min + 1e-5)
    return norm_flatten_score



class PatchNet(nn.Module):
    def __init__(self, score, k, in_channels, stride=None, num_samples=500):
        super(PatchNet, self).__init__()
        self.k = k
        self.anchor_size = int(sqrt(k))
        self.stride = stride
        self.score = score
        self.in_channels = in_channels
        self.num_samples = num_samples

        if score == 'tpool':
            self.score_network = PredictorLG(embed_dim=2*in_channels)
        
        elif score == 'spatch':
            self.score_network = PredictorLG(embed_dim=in_channels)
            self.init = torch.eye(self.k).unsqueeze(0).unsqueeze(-1).cuda()
    
    def get_indicator(self, scores, k, sigma):
        indicator = PerturbedTopKFunction.apply(scores, k, self.num_samples, sigma)
        indicator = einops.rearrange(indicator, "b k d -> b d k")
        return indicator
    
    def get_indices(self, scores, k):
        indices = HardTopK(k, scores)
        return indices
    
    def generate_random_indices(self, b, n, k):
        indices = []
        for _ in range(b):
            indice = np.sort(np.random.choice(n, k, replace=False))
            indices.append(indice)
        indices = np.vstack(indices)
        indices = torch.Tensor(indices).long().cuda()
        return indices
    
    def generate_uniform_indices(self, b, n, k):
        indices = torch.linspace(0, n-1, steps=k).long()
        indices = indices.unsqueeze(0).cuda()
        indices = indices.repeat(b, 1)
        return indices


    def forward(self, x, type, N, T, sigma):
        B = x.size(0)
        H = W = int(sqrt(N))
        indicator = None
        indices = None

        if type == 'time':
            if self.score == 'tpool':
                x = rearrange(x, 'b (t n) m -> b t n m', t=T)
                avg = torch.mean(x, dim=2, keepdim=False)
                max_ = torch.max(x, dim=2).values
                x_ = torch.cat((avg, max_), dim=2)
                scores = self.score_network(x_).squeeze(-1)
                scores = min_max_norm(scores)
                
                if self.training:
                    indicator = self.get_indicator(scores, self.k, sigma)
                else:
                    indices = self.get_indices(scores, self.k)
                x = rearrange(x, 'b t n m -> b t (n m)')
            
        else:
            s = self.stride if self.stride is not None else int(max((H - self.anchor_size) // 2, 1))
          
            if self.score == 'spatch':
                x = rearrange(x, 'b (t n) c -> (b t) n c', t=T)
                scores = self.score_network(x)
                scores = rearrange(scores, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                scores = F.unfold(scores, kernel_size=self.anchor_size, stride=s)
                scores = scores.mean(dim=1)
                scores = min_max_norm(scores)
                
                x = rearrange(x, '(b t) (h w) c -> (b t) c h w', b=B, h=H)
                x = F.unfold(x, kernel_size=self.anchor_size, stride=s).permute(0, 2, 1).contiguous()

                if self.training:
                    indicator = self.get_indicator(scores, 1, sigma)
                    
                else:
                    indices = self.get_indices(scores, 1)
         
            

        
        if self.training:
            if indicator is not None:
                patches = extract_patches_from_indicators(x, indicator)

            elif indices is not None:
                patches = extract_patches_from_indices(x, indices)
                
            if type == 'time':
                patches = rearrange(patches, 'b k (n c) -> b (k n) c', n = N)

            elif self.score == 'spatch':
                patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c',
                    b=B, c=self.in_channels, kh=self.anchor_size) 

            return patches
        
        
        else:
            patches = extract_patches_from_indices(x, indices)
            
            if type == 'time':
                patches = rearrange(patches, 'b k (n c) -> b (k n) c', n = N)

            elif self.score == 'spatch':
                patches = rearrange(patches, '(b t) k (c kh kw) -> b (t k kh kw) c', 
                    b=B, c=self.in_channels, kh=self.anchor_size)
            
            return patches
            
            

