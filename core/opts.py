import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import einsum
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

class cross_attention(nn.Module):
    def __init__(self, in_dim, out_dim, head) -> None:
        super(self, cross_attention).__init__()
        
        self.qurey = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.outlayers = nn.Linear(out_dim, in_dim)
        self.h_size = out_dim // in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = head
        self.scale = out_dim ** (0.5)
        
    def forward(self, x, y, mask=None):
        batch_size, c, h, w = x.size()
        q = self.qurey(y).view(batch_size, -1, self.heads, self.h_size).transpose(1,2)
        k = self.key(x).view(batch_size, -1, self.heads, self.h_size).transpose(1,2)
        v = self.value(x).view(batch_size, -1, self.heads, self.h_size).transpose(1,2)
        
        attention = torch.matmul(q,torch.transpose(k, -1, -2)) 
        attention = torch.matmul(nn.Softmax(attention / torch.sqrt(q.size(-1)), dim=-1), v)
        
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.h_size)
        
        return self.outlayers(attention)      
        
        
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        if context != None:
            context = context
        else:
            context = x
        
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)       
        
class SpitalCrossAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()        
        self.in_channel = in_channel
        self.act = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(in_channel)
        # self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channel, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels=in_channel,
                           out_channels=in_channel,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.k = nn.Conv2d(in_channels=in_channel,
                           out_channels=in_channel,
                           kernel_size=1,
                           stride=1,
                           padding=0)    
        self.v = nn.Conv2d(in_channels=in_channel,
                           out_channels=in_channel,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.out_layer = nn.Conv2d(in_channels=in_channel,
                                   out_channels=in_channel,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        
    def forward(self, x, Q):
        identity = x
        q = self.q(Q)
        k = self.k(x)
        v = self.v(x)
        
        b, c, h, w = q.size()
        scale = int(c) ** -0.5
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')
        
        score = nn.functional.softmax(einsum('b i j, b j k -> b i k', q, k) * scale)
        score = rearrange(score, 'b i j -> b j i')
        
        v_ = einsum('b i j, b j k -> b i k', v, score)
        v_ = rearrange(v_, 'b c (h w) -> b c h w', h=h)
        v_ = self.out_layer(v_)
        
        return v_
    
class self_attention(nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.ouotput_layer = nn.Linear(out_dim, in_dim)
        self.scale = in_dim ** -0.5
        
    def forward(self, x, mask=None):
        b, c, h, w = x.size()
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q = rearrange(q, 'b n (h d) -> b (h d) n', n=self.heads)
        k = rearrange(k, 'b n (h d) -> b n (h d)', n=self.heads)
        v = rearrange(v, 'b n (h d) -> b (h d) n', n=self.heads)
        score = einsum('bij, bjk -> bik', q, k)
        score = nn.Softmax(score * self.scale, dim=-1)
        
        out = einsum('bij, bjk -> bik', score, v)
        out = rearrange('b (h d) n -> b n (h d)', out, n=self.heads)
        out = self.ouotput_layer(out)
        return out
    
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
    

class motion_att(nn.Module):
    def __init__(self, in_channels, *args, **kargs):
        super(motion_att).__init__()

        self.in_channel = in_channels

        self.conv1 = nn.Conv2d(self.in_channel, self.in_channel, (1, 11), (1, 1), (0, 1))
        self.conv2 = nn.Conv2d(self.in_channel, self.in_channel, (11, 1), (1, 1), (1, 0))
        self.conv3 = nn.Conv2d(2 * self.in_channel, self.in_channel, (1, 1), (1, 1))
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x1 = self.bn(self.conv1(x))
        x1 = self.act(x1)
        x1 = self.bn(self.conv1(x1))
        x1 = self.sigmoid(x1)
        x1 = x * x1

        x2 = self.bn(self.conv2(x))
        x2 = self.act(x2)
        x2 = self.bn(self.conv2(x2))
        x2 = self.sigmoid(x2)
        x2 = x * x2

        output = self.conv3(torch.cat(x1, x2, dim=1))

        return output


class cross_model(nn.Module):
    def __init__(self, in_chanels, dim, heads):
        super().__init__()
        
        self.inc = in_chanels
        self.dim = dim
        self.heads = heads
        self.block = SpitalCrossAttention(self.inc)
        
    def forward(self, x, y):
        
        v_ = self.block(x, y)
        return v_
        
        
        
        