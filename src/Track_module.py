import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.utils.checkpoint as checkpoint
from util_module import *
from Attention_module import *

# Components for two-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair)
# 2. Pair -> Pair update (Triangle-multiplicative updates & axial attention)
# 3. MSA -> Pair update (extract coevolution signal)

# Module contains classes and functions to generate initial embeddings
class SeqSep(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32):
        super(SeqSep, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
    
    def forward(self, idx, oligo=1):
        B, L = idx.shape[:2]
        bins = torch.arange(self.minpos, self.maxpos, device=idx.device)
        seqsep = torch.full((oligo,L,L), 100, dtype=idx.dtype, device=idx.device)
        seqsep[0] = idx[:,None,:] - idx[:,:,None] # (B, L, L)
        #
        ib = torch.bucketize(seqsep, bins).long() # (B, L, L)
        emb = self.emb(ib) #(B, L, L, d_model)
        return emb

# Update MSA with biased self-attention. bias from Pair
class MSAPair2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8,
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(MSAPair2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.drop_row = CustomDropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
    def forward(self, msa, pair):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L, _ = msa.shape

        pair = self.norm_pair(pair)
        
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa

class Pair2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, p_drop=0.25):
        super(Pair2Pair, self).__init__()
        
        self.drop_row = CustomDropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = CustomDropout(broadcast_dim=2, p_drop=p_drop)
        
        self.tri_mul_out = TriangleMultiplication(d_pair, d_hidden=d_hidden)
        self.tri_mul_in = TriangleMultiplication(d_pair, d_hidden, outgoing=False)

        self.ff = FeedForwardLayer(d_pair, 2)
    
    def forward(self, pair):
        #_nc = lambda x:torch.sum(torch.isnan(x))
        pair = pair + self.drop_row(self.tri_mul_out(pair)) 
        pair = pair + self.drop_col(self.tri_mul_in(pair)) 
        pair = pair + self.ff(pair)

        return pair

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        self.d_hidden = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)

        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
       
        pair = pair + out
        
        return pair

class IterBlock(nn.Module):
    def __init__(self, d_msa=256, d_pair=128,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, p_drop=0.15):
        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        self.msa2msa = MSAPair2MSA(d_msa=d_msa, d_pair=d_pair,
                                   n_head=n_head_msa,
                                   use_global_attn=use_global_attn,
                                   d_hidden=d_hidden_msa, p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                 d_hidden=d_hidden//2, p_drop=p_drop)
        self.pair2pair = Pair2Pair(d_pair=d_pair, n_head=n_head_pair, 
                                   d_hidden=d_hidden, p_drop=p_drop)

    def forward(self, msa, pair, use_checkpoint=False):
        if use_checkpoint:
            msa = checkpoint.checkpoint(create_custom_forward(self.msa2msa), msa, pair)
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair)
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair)
        else:
            msa = self.msa2msa(msa, pair)
            pair = self.msa2pair(msa, pair)
            pair = self.pair2pair(pair)
        
        return msa, pair

class IterativeSimulator(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12,
                 d_msa=256, d_msa_extra=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 p_drop=0.15):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        
        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.ModuleList([IterBlock(d_msa=d_msa_extra, d_pair=d_pair,
                                                        n_head_msa=n_head_msa,
                                                        n_head_pair=n_head_pair,
                                                        d_hidden_msa=8,
                                                        d_hidden=d_hidden,
                                                        p_drop=p_drop,
                                                        use_global_attn=True)
                                                        for i in range(n_extra_block)])

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.ModuleList([IterBlock(d_msa=d_msa, d_pair=d_pair,
                                                       n_head_msa=n_head_msa,
                                                       n_head_pair=n_head_pair,
                                                       d_hidden=d_hidden,
                                                       p_drop=p_drop,
                                                       use_global_attn=False)
                                                       for i in range(n_main_block)])


    def forward(self, msa, msa_extra, pair, use_checkpoint=False):
        # input:
        #   msa: seed MSA embeddings (B, N, L, d_msa)
        #   msa_extra: extra MSA embeddings (B, N, L, d_msa_extra)
        #   pair: initial residue pair embeddings (B, L, L, d_pair)
        #   use_checkpoint: use gradient checkpointing or not to save memory usage during training

        for i_m in range(self.n_extra_block):
            msa_extra, pair = self.extra_block[i_m](msa_extra, pair, use_checkpoint=use_checkpoint)

        for i_m in range(self.n_main_block):
            msa, pair = self.main_block[i_m](msa, pair, use_checkpoint=use_checkpoint)
       
        return msa, pair
