import torch
import torch.nn as nn
from Embeddings import MSA_emb, Extra_emb, Templ_emb, Recycling
from Track_module import IterativeSimulator
from AuxiliaryPredictor import DistanceNetwork, MaskedTokenNetwork, ExpResolvedNetwork
from util import INIT_CRDS
from torch import einsum

class RF2trackModule(nn.Module):
    def __init__(self, n_extra_block=4, n_main_block=12,
                 d_msa=256, d_msa_extra=64, d_pair=128, d_templ=64,
                 n_head_msa=8, n_head_pair=4, n_head_templ=4,
                 d_hidden=32, d_hidden_templ=32,
                 p_drop=0.15):
        super(RF2trackModule, self).__init__()
        #
        # Input Embeddings
        self.seed_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, p_drop=p_drop)
        self.extra_emb = Extra_emb(d_msa=d_msa_extra, d_init=23, p_drop=p_drop)
        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ,
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25)
        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair)
        #
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            d_msa=d_msa, d_msa_extra=d_msa_extra,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.exp_pred = ExpResolvedNetwork(d_msa)

    def forward(self, msa_seed=None, msa_extra=None, seq=None, idx=None,
                t1d=None, t2d=None, same_chain=None, 
                msa_prev=None, pair_prev=None,
                return_raw=False, 
                use_checkpoint=False):

        B, N, L = msa_seed.shape[:3]
        dtype = msa_seed.dtype

        # Get embeddings
        msa_seed, pair = self.seed_emb(msa_seed, seq, idx, same_chain)
        msa_extra = self.extra_emb(msa_extra, seq, idx)
        msa_seed, pair = msa_seed.to(dtype), pair.to(dtype)
        msa_extra = msa_extra.to(dtype)

        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_seed[:,0])
            pair_prev = torch.zeros_like(pair)
        msa_seed, pair = self.recycle(msa_seed, pair, msa_prev, pair_prev)
        # add template embedding
        pair = self.templ_emb(t1d, t2d, pair, use_checkpoint=use_checkpoint)
        
        # Predict coordinates from given inputs
        msa, pair = self.simulator(msa_seed, msa_extra, pair, use_checkpoint=use_checkpoint)

        if return_raw:
            return msa[:,0], pair

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)
        # predict distogram & orientograms
        logits = self.c6d_pred(pair)

        # predict experimentally resolved or not
        logits_exp = self.exp_pred(msa[:,0])
        
        return logits, logits_aa, logits_exp
