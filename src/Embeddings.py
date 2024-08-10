import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.utils.checkpoint as checkpoint
from util_module import create_custom_forward, init_lecun_normal
from Attention_module import Attention, FeedForwardLayer, AttentionWithBias
from Track_module import Pair2Pair

# Module contains classes and functions to generate initial embeddings
class PositionalEncoding2D(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-31, maxpos=31):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin+1, d_model)
        self.emb_chain = nn.Embedding(2, d_model)
        self.project = nn.Linear(d_model*2, d_model, bias=False)
    
    def forward(self, idx, same_chain):
        B, L = idx.shape[:2]

        inter_idx = torch.full((B,L,L), self.nbin, device=idx.device)
        seqsep = idx[:,None,:] - idx[:,:,None] + self.maxpos # (B, L, L)
        seqsep = torch.clamp(seqsep, min=0, max=(self.nbin-1))
        seqsep = seqsep * same_chain + inter_idx * (1-same_chain)
        #
        emb_residx = self.emb(seqsep) #(B, L, L, d_model)
        emb_chain  = self.emb_chain(same_chain.long())
        #
        emb = self.project(torch.cat((emb_residx, emb_chain), dim=-1))
        return emb

class MSA_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_pair=128, d_init=23+23,
                 minpos=-31, maxpos=31, p_drop=0.1):
        super(MSA_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(23, d_msa) # embedding for query sequence -- used for MSA embedding
        self.emb_left = nn.Embedding(23, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(23, d_pair) # embedding for query sequence -- used for pair embedding
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos)

        self.d_init = d_init
        self.d_msa = d_msa

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)

        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, same_chain):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        #   - same_chain: whether two residues are in the same_chain
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #   - pair: Initial Pair embedding (B, L, L, d_pair)

        B, N, L = msa.shape[:3] # number of sequenes in MSA

        # msa embedding 
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        tmp = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + tmp.expand(-1, N, -1, -1) # adding query embedding to MSA

        # pair embedding 
        left = self.emb_left(seq)[:,None] # (B, 1, L, d_pair)
        right = self.emb_right(seq)[:,:,None] # (B, L, 1, d_pair)
        pair = (left + right) # (B, L, L, d_pair)
        pair = pair + self.pos(idx, same_chain) # add relative position

        return msa, pair

class Extra_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_init=23, p_drop=0.1):
        super(Extra_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(23, d_msa) # embedding for query sequence

        self.d_init = d_init
        self.d_msa = d_msa

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #N = msa.shape[1] # number of sequenes in MSA

        B,N,L = msa.shape[:3]
        
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        seq = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA

        return msa 

# TODO
class TemplatePairStack(nn.Module):
    # process template pairwise features
    # use structure-biased attention
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.25):
        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        proc_s = [Pair2Pair(d_pair=d_templ, n_head=n_head, d_hidden=d_hidden, p_drop=p_drop) for i in range(n_block)]
        self.block = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)

    def forward(self, templ, use_checkpoint=False):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B*T, L, L, -1)

        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = checkpoint.checkpoint(create_custom_forward(self.block[i_block]), templ) 
            else:
                templ = self.block[i_block](templ)
        return self.norm(templ).reshape(B, T, L, L, -1)

class Templ_emb(nn.Module):
    # Get template embedding
    # Features are
    #   t2d:
    #   - 37 distogram bins + 6 orientations (43)
    #   - Mask (missing/unaligned) (1)
    #   t1d:
    #   - tiled AA sequence (20 standard aa + gap)
    #   - confidence (1)
    #   
    def __init__(self, d_t1d=21+1, d_t2d=43+1, d_pair=128,
                 n_block=2, d_templ=64,
                 n_head=4, d_hidden=16, p_drop=0.25):
        super(Templ_emb, self).__init__()
        # process 2D features
        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ)
        self.templ_stack = TemplatePairStack(n_block=n_block,
                                             d_templ=d_templ, n_head=n_head,
                                             d_hidden=d_hidden, p_drop=p_drop)
        
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair, p_drop=p_drop)

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)
    
    def _get_templ_emb(self, t1d, t2d):
        B, T, L, _ = t1d.shape
        # Prepare 2D template features
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        #
        templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, 88)
        return self.emb(templ) # Template templures (B, T, L, L, d_templ)
    
    def forward(self, t1d, t2d, pair, use_checkpoint=False):
        # Input
        #   - t1d: 1D template info (B, T, L, 22)
        #   - t2d: 2D template info (B, T, L, L, 44)
        #   - pair: query pair features (B, L, L, d_pair)
        
        B, T, L, _ = t1d.shape
        
        templ = self._get_templ_emb(t1d, t2d)
        
        # process each template pair feature
        templ = self.templ_stack(templ, use_checkpoint=use_checkpoint)

        # mixing query pair features to template information (Template pointwise attention)
        pair = pair.reshape(B*L*L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B*L*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn), pair, templ, templ)
            out = out.reshape(B, L, L, -1)
        else:
            out = self.attn(pair, templ, templ).reshape(B, L, L, -1)
        #
        pair = pair.reshape(B, L, L, -1)
        pair = pair + out

        return pair

class Recycling(nn.Module):
    def __init__(self, d_msa=256, d_pair=128):
        super(Recycling, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)

    def forward(self, msa, pair, msa_prev, pair_prev):
        B, L = msa.shape[:2]
        msa[:,0] = msa[:,0] + self.norm_msa(msa_prev)
        pair = pair + self.norm_pair(pair_prev)

        return msa, pair

