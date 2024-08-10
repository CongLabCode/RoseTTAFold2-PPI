import os, csv, torch
import numpy as np
from torch.utils import data
from dateutil import parser
from parsers import parse_a3m, parse_pdb, parse_pdb_w_miss, parse_pdb_w_plddt
from chemical import INIT_CRDS
from util import center_and_realign_missing, random_rot_trans

base_dir = "/projects/ml/TrRosetta/PDB-2021AUG02"
fb_dir = "/projects/ml/TrRosetta/fb_af"
compl_dir = "/net/tukwila/congq/PDB_complex"
muld_dir = "/net/tukwila/congq/AFDB_multidom"


# optional: exclude homologs to the validation set in FB and MULD set
def set_data_loader_params(args):
	PARAMS = {
		"COMPL_LIST" : "%s/list.nohomo.csv"%compl_dir,
		"NEGATIVE_LIST"	: "%s/list.nohomo.negative.csv"%compl_dir,
		"PDB_LIST"	 : "%s/list_v02.csv"%base_dir,
		"FB_LIST"	 : "%s/list_b1-3.csv"%fb_dir,
		"MULD_LIST"	 : "%s/list.csv"%muld_dir,
		"MULD_NEG_LIST"	: "%s/list.negative.csv"%muld_dir,
		"VAL_PDB"	 : "%s/val/xaa"%base_dir,
		"VAL_COMPL"  : "%s/list.val"%compl_dir,
		"VAL_NEG"	 : "%s/list.val.negative"%compl_dir,
		"PDB_DIR"	 : base_dir,
		"FB_DIR"	 : fb_dir,
		"COMPL_DIR"  : compl_dir,
		"MULD_DIR"	 : muld_dir,
		"MINTPLT" : 0,
		"MAXTPLT" : 0,
		"MINSEQ"  : 1,
		"MAXSEQ"  : 1024,
		"MAXLAT"  : 128, 
		"CROP"	  : 256,
		"DATCUT"  : "2024-AUG-30",
		"RESCUT"  : 5.0,
		"BLOCKCUT": 5,
		"PLDDTCUT": 70.0,
		"ROWS"	  : 1,
		"MAXCYCLE": 3
	}

	for param in PARAMS:
		if hasattr(args, param.lower()):
			PARAMS[param] = getattr(args, param.lower())
	return PARAMS


def MSABlockDeletion(msa, nb=5):
	'''
	Down-sample given MSA by randomly delete blocks of sequences
	Input: MSA/Insertion having shape (N, L)
	output: new MSA/Insertion with block deletion (N', L)
	'''
	# select 5 start size and delete upto 10% since each start set.
	# overall about 25% of sequences will be deleted.
	N, L = msa.shape
	block_size = max(int(N*0.1), 1)
	block_start = np.random.randint(low=1, high=N, size=nb)
	to_delete = block_start[:,None] + np.arange(block_size)[None,:]
	to_delete = np.unique(np.clip(to_delete, 1, N-1))
	mask = np.ones(N, bool)
	mask[to_delete] = 0
	return msa[mask]


def cluster_sum(data, assignment, B, N_seq, N_res):
	# Get statistics from clustering results (clustering extra sequences with seed sequences)
	csum = torch.zeros((B, N_seq, N_res, data.shape[-1]), device=data.device).scatter_add(1, \
					assignment.view(B,-1,1,1).expand(B,-1,N_res,data.shape[-1]), data.float())
	return csum

def MSAFeaturize(msa, params, p_mask=0.15, eps=1e-6):
	'''
	Input: full MSA information (after Block deletion if necessary)
	Output: seed MSA features & extra MSA features	
	Seed MSA features:
		- aatype of seed sequence (20 regular aa + 1 unknown + 1 gap + 1 mask)
		- profile of clustered sequences (23)
	extra sequence features:
		- aatype of extra sequence (23)
	'''
	B, N, L = msa.shape
	# raw MSA profile
	raw_profile = torch.nn.functional.one_hot(msa, num_classes=22)
	raw_profile = raw_profile.float().mean(dim=1) # (B, L, 22)

	# N_seed sequences will be selected randomly as a seed MSA (aka latent MSA)
	# - First sequence is always query sequence
	# - the rest of sequences are selected randomly
	N_seed = min(N, params['MAXLAT'])
	sample = torch.randperm(N-1, device=msa.device)
	msa_seed = torch.cat((msa[:,:1,:], msa[:,1:,:][:,sample[:N_seed-1]]), dim=1)

	# 15% random masking 
	# - 10%: aa replaced with a uniformly sampled random amino acid
	# - 10%: aa replaced with an amino acid sampled from the MSA profile
	# - 10%: not replaced
	# - 70%: replaced with a special token ("mask")
	random_aa = torch.tensor([[0.05]*20 + [0.0]*2], device=msa.device)
	same_aa = torch.nn.functional.one_hot(msa_seed, num_classes=22)
	probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
	probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
		
	sampler = torch.distributions.categorical.Categorical(probs=probs)
	mask_sample = sampler.sample()
	mask_pos = torch.rand(msa_seed.shape, device=msa_seed.device) < p_mask
	msa_masked = torch.where(mask_pos, mask_sample, msa_seed)
	seq = msa_masked[:,0].clone()
	   
	# get extra sequenes
	if N - N_seed > 0: # There is at least one extra sequence
		msa_extra = torch.cat((msa_masked[:,:1,:], msa[:,1:,:][:,sample[N_seed-1:]]), dim=1)[:,:params['MAXSEQ']]
		extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
		extra_mask[:,0] = mask_pos[:,0].clone()
	else:
		msa_extra = msa_masked[:,:1].clone() # query as the only extra sequence
		extra_mask = mask_pos[:,:1].clone()
	N_extra = msa_extra.shape[1]
		
	# 1. one_hot encoded aatype: msa_clust_onehot
	msa_seed_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=23) #(B,N_seed,L,23)
	msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=23) #(B,N_extra,L,23)
		
	# clustering (assign remaining sequences to their closest cluster by Hamming distance
	count_seed = torch.logical_and(~mask_pos, msa_seed != 21) # 21 index for gap, ignore both masked & gaps
	count_extra = torch.logical_and(~extra_mask, msa_extra != 21) 

	# get number of identical tokens for each pair of sequences (extra vs seed)
	agreement = torch.matmul(
			(count_extra[:,:,:,None]*msa_extra_onehot).float().view(B, N_extra, -1), 
			(count_seed[:,:,:,None]*msa_seed_onehot).float().view(B, N_seed, -1).permute(0,2,1)) 
	# (B, N_extra, N_seed)
	assignment = torch.argmax(agreement, dim=-1) # map each extra seq to the closest seed seq

	# 2. cluster profile -- ignore masked token when calculate profiles
	count_extra = ~extra_mask # only consider non-masked tokens in extra seqs
	count_seed = ~mask_pos # only consider non-masked tokens in seed seqs
	msa_seed_profile = cluster_sum(count_extra[:,:,:,None] * msa_extra_onehot, assignment, B, N_seed, L)
	msa_seed_profile += count_seed[:,:,:,None] * msa_seed_onehot
	count_profile = cluster_sum(count_extra[:,:,:,None], assignment, B, N_seed, L).view(B, N_seed, L)
	count_profile += count_seed
	count_profile += eps
	msa_seed_profile /= count_profile[:,:,:,None]

	# seed MSA features (one-hot aa, cluster profile, ins statistics, terminal info)
	msa_seed_feat = torch.cat((msa_seed_onehot, msa_seed_profile), dim=-1)
	return seq, msa_seed_feat, msa_extra_onehot.float(), mask_pos, msa_seed


def get_train_valid_set(params):
	# read validation IDs for PDB set
	val_pdb_ids = set([int(l) for l in open(params['VAL_PDB']).readlines()])
	val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])
	val_neg_ids = set([int(l) for l in open(params['VAL_NEG']).readlines()])
	
	# read & clean list.csv
	# rows = [pdbchain, hash, cluster, length]
	with open(params['PDB_LIST'], 'r') as f:
		reader = csv.reader(f)
		next(reader)
		rows = [[r[0], r[3], int(r[4]), int(r[-1].strip())] for r in reader if float(r[2]) <= \
						params['RESCUT'] and parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

	# compile training and validation sets
	# train_pdb[cluster] = [([pdbchain, hash], length)]
	# valid_pdb[cluster] = [([pdbchain, hash], length)]
	val_ids = list()
	train_pdb = {}
	valid_pdb = {}
	for r in rows:
		if r[2] in val_pdb_ids:
			val_ids.append(r[0])
			if r[2] in valid_pdb.keys():
				valid_pdb[r[2]].append((r[:2], r[-1]))
			else:
				valid_pdb[r[2]] = [(r[:2], r[-1])]
		else:
			if r[2] in train_pdb.keys():
				train_pdb[r[2]].append((r[:2], r[-1]))
			else:
				train_pdb[r[2]] = [(r[:2], r[-1])]
	val_ids = set(val_ids)
	
	# compile facebook model sets
	# rows = [unirefid, hash, cluster, length]
	# fb[cluster] = [([unirefid, hash], length)]
	with open(params['FB_LIST'], 'r') as f:
		reader = csv.reader(f)
		next(reader)
		rows = [[r[0], r[2], int(r[3]), len(r[-1].strip())] for r in reader if float(r[1]) > 80.0 \
						and len(r[-1].strip()) > 200] # requires pLDDT above 80 and above 200 residues
	fb = {}
	for r in rows:
		if r[2] in fb.keys():
			fb[r[2]].append((r[:2], r[-1]))
		else:
			fb[r[2]] = [(r[:2], r[-1])]

	# compile multi-domain sets
	# rows = [pairid, hash, cluster, [len1, len2]]
	# muld[cluster] = [([pairid, hash], [len1, len2])]
	with open(params['MULD_LIST'], 'r') as f:
		reader = csv.reader(f)
		next(reader)
		rows = [[r[0], r[1], int(r[2]), [int(plen) for plen in r[3].split(':')]] for r in reader]
	muld = {}
	for r in rows:
		if r[2] in muld.keys():
			muld[r[2]].append((r[:2], r[-1]))
		else:
			muld[r[2]] = [(r[:2], r[-1])]

	with open(params['MULD_NEG_LIST'], 'r') as f:
		reader = csv.reader(f)
		next(reader)
		rows = [[r[0], r[1], int(r[2]), [int(plen) for plen in r[3].split(':')]] for r in reader]
	muld_neg = {}
	for r in rows:
		if r[2] in muld_neg.keys():
			muld_neg[r[2]].append((r[:2], r[-1]))
		else:
			muld_neg[r[2]] = [(r[:2], r[-1])]

	# compile complex sets
	# rows = [pairid, hash, cluster, [len1, len2]]
	with open(params['COMPL_LIST'], 'r') as f:
		reader = csv.reader(f)
		next(reader)
		rows = [[r[0], r[1], int(r[2]), [int(plen) for plen in r[3].split(':')]] for r in reader]

	# train_compl[cluster] = [([pairid, hash], [len1, len2])]
	# valid_compl[cluster] = [([pairid, hash], [len1, len2])]
	train_compl = {}
	valid_compl = {}
	for r in rows:
		if r[2] in val_compl_ids:
			if r[2] in valid_compl.keys():
				valid_compl[r[2]].append((r[:2], r[-1]))
			else:
				valid_compl[r[2]] = [(r[:2], r[-1])]
		else:
			# if subunits are included in PDB validation set, exclude them from training
			pdbA, pdbB = r[0].split(':')
			if pdbA in val_ids:
				continue
			if pdbB in val_ids:
				continue
			if r[2] in train_compl.keys():
				train_compl[r[2]].append((r[:2], r[-1]))
			else:
				train_compl[r[2]] = [(r[:2], r[-1])]

	# compile negative examples
	# remove pairs if any of the subunits are included in validation set
	# rows = [pairid, hash, cluster, [len1, len2]]
	with open(params['NEGATIVE_LIST'], 'r') as f:
		reader = csv.reader(f)
		next(reader)
		rows = [[r[0], r[1], int(r[2]), [int(plen) for plen in r[3].split(':')]] for r in reader]

	# train_neg[cluster] = [([pairid, hash], [len1, len2])]
	# valid_neg[cluster] = [([pairid, hash], [len1, len2])]
	train_neg = {}
	valid_neg = {}
	for r in rows:
		if r[2] in val_neg_ids:
			if r[2] in valid_neg.keys():
				valid_neg[r[2]].append((r[:2], r[-1]))
			else:
				valid_neg[r[2]] = [(r[:2], r[-1])]
		else:
			pdbA, pdbB = r[0].split(':')
			if pdbA in val_ids:
				continue
			if pdbB in val_ids:
				continue
			if r[2] in train_neg.keys():
				train_neg[r[2]].append((r[:2], r[-1]))
			else:
				train_neg[r[2]] = [(r[:2], r[-1])]
	
	# Get average chain length in each cluster and calculate weights
	pdb_IDs = list(train_pdb.keys())
	fb_IDs = list(fb.keys())
	muld_IDs = list(muld.keys())
	muld_neg_IDs = list(muld_neg.keys())
	compl_IDs = list(train_compl.keys())
	neg_IDs = list(train_neg.keys())

	pdb_weights = list()
	fb_weights = list()
	muld_weights = list()
	muld_neg_weights = list()
	compl_weights = list()
	neg_weights = list()

	for key in pdb_IDs:
		plen = sum([plen for _, plen in train_pdb[key]]) // len(train_pdb[key])
		w = (1/512.)*max(min(float(plen),512.),256.)
		pdb_weights.append(w)
	
	for key in fb_IDs:
		plen = sum([plen for _, plen in fb[key]]) // len(fb[key])
		w = (1/512.)*max(min(float(plen),512.),256.)
		fb_weights.append(w)
	
	for key in muld_IDs:
		plen = sum([sum(plen) for _, plen in muld[key]]) // len(muld[key])
		w = (1/512.)*max(min(float(plen),512.),256.)
		muld_weights.append(w)

	for key in muld_neg_IDs:
		plen = sum([sum(plen) for _, plen in muld_neg[key]]) // len(muld_neg[key])
		w = (1/512.)*max(min(float(plen),512.),256.)
		muld_neg_weights.append(w)

	for key in compl_IDs:
		plen = sum([sum(plen) for _, plen, in train_compl[key]]) // len(train_compl[key])
		w = (1/512.)*max(min(float(plen),512.),256.)
		compl_weights.append(w)
	
	for key in neg_IDs:
		plen = sum([sum(plen) for _, plen, in train_neg[key]]) // len(train_neg[key])
		w = (1/512.)*max(min(float(plen),512.),256.)
		neg_weights.append(w)

	return (pdb_IDs, torch.tensor(pdb_weights).float(), train_pdb), \
		   (fb_IDs, torch.tensor(fb_weights).float(), fb), \
		   (muld_IDs, torch.tensor(muld_weights).float(), muld), \
		   (muld_neg_IDs, torch.tensor(muld_neg_weights).float(), muld_neg), \
		   (compl_IDs, torch.tensor(compl_weights).float(), train_compl), \
		   (neg_IDs, torch.tensor(neg_weights).float(), train_neg), \
		   valid_pdb, valid_compl, valid_neg


# slice long chains
def get_crop(l, mask, device, params):
	sel = torch.arange(l,device=device)
	if l <= params['CROP']:
		return sel
	size = params['CROP']
	mask = ~(mask[:,:3].sum(dim=-1) < 3.0)
	exists = mask.nonzero()[0]

	res_idx = exists[torch.randperm(len(exists))[0]].item()
	lower_bound = max(0, res_idx-size+1)
	upper_bound = min(l-size, res_idx+1)
	start = np.random.randint(lower_bound, upper_bound)
	return sel[start:start+size]

def get_complex_crop(len_s, mask, device, params):
	tot_len = sum(len_s)
	sel = torch.arange(tot_len, device=device)
	n_added = 0
	n_remaining = sum(len_s)
	preset = 0
	sel_s = list()

	for k in range(len(len_s)):
		n_remaining -= len_s[k]
		crop_max = min(params['CROP']-n_added, len_s[k])
		crop_min = min(len_s[k], max(1, params['CROP'] - n_added - n_remaining))		
		if k == 0:
			crop_max = min(crop_max, params['CROP']-5)
		crop_size = np.random.randint(crop_min, crop_max+1)
		n_added += crop_size
		
		mask_chain = ~(mask[preset:preset+len_s[k],:3].sum(dim=-1) < 3.0)
		exists = mask_chain.nonzero()[0]
		res_idx = exists[torch.randperm(len(exists))[0]].item()
		lower_bound = max(0, res_idx - crop_size + 1)
		upper_bound = min(len_s[k] - crop_size, res_idx) + 1
		start = np.random.randint(lower_bound, upper_bound) + preset
		sel_s.append(sel[start:start+crop_size])
		preset += len_s[k]
	return torch.cat(sel_s)

def get_spatial_crop(xyz, mask, sel, len_s, params, label, cutoff=10.0, eps=1e-6):
	device = xyz.device	
	# get interface residue
	cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff
	cond = torch.logical_and(cond, mask[:len_s[0],None,1] * mask[None,len_s[0]:,1]) 
	i,j = torch.where(cond)
	ifaces = torch.cat([i,j+len_s[0]])
	if len(ifaces) < 1:
		print ("ERROR: no iface residue????", label)
		return get_complex_crop(len_s, mask, device, params)

	# randomly select an interface residue and get the residues nearby. 
	cnt_idx = ifaces[np.random.randint(len(ifaces))]
	dist = torch.cdist(xyz[:,1], xyz[cnt_idx,1][None]).reshape(-1) + \
					torch.arange(len(xyz), device=xyz.device)*eps # add random noise to avoid ties
	cond = mask[:,1]*mask[cnt_idx,1]
	dist[~cond] = 999999.9
	_, idx = torch.topk(dist, params['CROP'], largest=False)
	sel, _ = torch.sort(sel[idx])
	return sel


# Generate input features for single-chain
def get_pdb(pdbfilename, plddtfilename, item, lddtcut):
	xyz, mask, res_idx = parse_pdb(pdbfilename)
	plddt = np.load(plddtfilename)
	mask = np.logical_and(mask, (plddt > lddtcut)[:,None])	
	return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx':torch.tensor(res_idx), 'label': item}

def get_pdb_miss(pdbfilename, L, item):
	xyz, mask, res_idx = parse_pdb_w_miss(pdbfilename, L)
	return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx':torch.tensor(res_idx), 'label':item}

def get_pdb_plddt(pdbfilename, item, lddtcut):
	xyz, mask, res_idx, plddt = parse_pdb_w_plddt(pdbfilename)
	mask = np.logical_and(mask, (plddt > lddtcut)[:,None])
	return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx':torch.tensor(res_idx), 'label':item}

def get_msa(a3mfilename, item, max_seq=10000):
	msa = parse_a3m(a3mfilename, max_seq=max_seq)
	return {'msa':torch.tensor(msa), 'label':item}


# Load PDB examples
def loader_pdb(item, params, random_noise=5.0):
	# load MSA, PDB, template info
	pdb = torch.load(params['PDB_DIR'] + '/torch/pdb/' + item[0][1:3] + '/' + item[0] + '.pt')
	a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])
   
	# get msa features
	msa = a3m['msa'].long()
	l_orig = msa.shape[1]
	if len(msa) > params['BLOCKCUT']:
		msa = MSABlockDeletion(msa)
 
	# get template features
	xyz_t = INIT_CRDS.reshape(1,1,27,3).repeat(1,l_orig,1,1) + torch.rand(1,l_orig,1,3)*random_noise
	f1d_t = torch.nn.functional.one_hot(torch.full((1, l_orig), 20).long(), num_classes=21).float()
	conf = torch.zeros((1, l_orig, 1)).float()
	f1d_t = torch.cat((f1d_t, conf), -1)
	mask_t = torch.full((1,l_orig,27), False)

	# get ground-truth structures
	idx = torch.arange(len(pdb['xyz']))
	xyz = INIT_CRDS.reshape(1, 27, 3).repeat(len(idx), 1, 1)
	xyz[:,:14,:] = pdb['xyz']
	mask = torch.full((len(idx), 27), False)
	mask[:,:14] = pdb['mask']
	xyz = torch.nan_to_num(xyz)

	# Residue cropping
	crop_idx = get_crop(len(idx), mask, mask.device, params)
	msa = msa[:,crop_idx]
	xyz_t = xyz_t[:,crop_idx]
	f1d_t = f1d_t[:,crop_idx]
	mask_t = mask_t[:,crop_idx]
	xyz = xyz[crop_idx]
	mask = mask[crop_idx]
	idx = idx[crop_idx]

	chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
	return msa.long(),xyz_t.float(),f1d_t.float(),mask_t,xyz.float(),mask,idx.long(),chain_idx,False,False


def loader_fb(item, params, random_noise=5.0):
	# loads sequence/structure/plddt information 
	a3m = get_msa(os.path.join(params["FB_DIR"],"a3m",item[-1][:2],item[-1][2:],item[0]+".a3m.gz"), item[0])
	pdb = get_pdb(os.path.join(params["FB_DIR"],"pdb",item[-1][:2],item[-1][2:],item[0]+".pdb"), \
					os.path.join(params["FB_DIR"],"pdb",item[-1][:2],item[-1][2:],item[0]+".plddt.npy"), \
					item[0], params['PLDDTCUT'])
	
	# get msa features
	msa = a3m['msa'].long()
	l_orig = msa.shape[1]
	if len(msa) > params['BLOCKCUT']:
		msa = MSABlockDeletion(msa)
	
	# get template features -- None
	xyz_t = INIT_CRDS.reshape(1,1,27,3).repeat(1,l_orig,1,1) + torch.rand(1,l_orig,1,3)*random_noise
	f1d_t = torch.nn.functional.one_hot(torch.full((1, l_orig), 20).long(), num_classes=21).float()
	conf = torch.zeros((1, l_orig, 1)).float()
	f1d_t = torch.cat((f1d_t, conf), -1)
	mask_t = torch.full((1,l_orig,27), False)
	
	# get PDB features
	idx = pdb['idx']
	xyz = INIT_CRDS.reshape(1,27,3).repeat(len(idx), 1, 1)
	xyz[:,:14,:] = pdb['xyz']
	mask = torch.full((len(idx), 27), False)
	mask[:,:14] = pdb['mask']

	# Residue cropping
	crop_idx = get_crop(len(idx), mask, mask.device, params)
	msa = msa[:,crop_idx]
	xyz_t = xyz_t[:,crop_idx]
	f1d_t = f1d_t[:,crop_idx]
	mask_t = mask_t[:,crop_idx]
	xyz = xyz[crop_idx]
	mask = mask[crop_idx]
	idx = idx[crop_idx]

	chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
	return msa.long(),xyz_t.float(),f1d_t.float(),mask_t,xyz.float(),mask,idx.long(),chain_idx,False,False


def loader_muld(item, L_s, params, negative=False, random_noise=5.0):
	dom_pair = item[0]
	domA, domB = item[0].split(':')
	hashA, hashB = item[1].split(':')
	a3m = {}
	if negative:
		pMSA_fn = params['MULD_DIR'] + '/nega_pmsas/' + hashA + '/' + domA + '__' + domB + '.i95c50.a3m'
	else:
		pMSA_fn = params['MULD_DIR'] + '/posi_pmsas/' + hashA + '/' + domA + '__' + domB + '.i95c50.a3m'
	a3m = get_msa(pMSA_fn, dom_pair)
	msa = a3m['msa']
	if len(msa) > params['BLOCKCUT']:
		msa = MSABlockDeletion(msa)

	# generate empty template vectors
	xyz_t_A = INIT_CRDS.reshape(1,1,27,3).repeat(1,L_s[0],1,1) + torch.rand(1,L_s[0],1,3)*random_noise
	f1d_t_A = torch.nn.functional.one_hot(torch.full((1, L_s[0]), 20).long(), num_classes=21).float()
	conf_A = torch.zeros((1, L_s[0], 1)).float()
	f1d_t_A = torch.cat((f1d_t_A, conf_A), -1)
	mask_t_A = torch.full((1, L_s[0], 27), False)

	xyz_t_B = INIT_CRDS.reshape(1,1,27,3).repeat(1,L_s[1],1,1) + torch.rand(1,L_s[1],1,3)*random_noise
	f1d_t_B = torch.nn.functional.one_hot(torch.full((1, L_s[1]), 20).long(), num_classes=21).float()
	conf_B = torch.zeros((1, L_s[1], 1)).float()
	f1d_t_B = torch.cat((f1d_t_B, conf_B), -1)
	mask_t_B = torch.full((1, L_s[1], 27), False)

	xyz_t = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B)), dim=1) # (T, L1+L2, natm, 3)
	f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=1) # (T, L1+L2, natm, 3)
	mask_t = torch.cat((mask_t_A, mask_t_B), dim=1) # (T, L1+L2, natm, 3)

	# get PDB features
	pdbA = get_pdb_plddt(params["MULD_DIR"] + '/dompdbs/' + hashA + '/' + domA + '.pdb', domA, params['PLDDTCUT'])
	pdbB = get_pdb_plddt(params["MULD_DIR"] + '/dompdbs/' + hashB + '/' + domB + '.pdb', domB, params['PLDDTCUT'])

	xyz = INIT_CRDS.reshape(1, 27, 3).repeat(sum(L_s), 1, 1)
	xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
	mask = torch.full((sum(L_s), 27), False)
	mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
	xyz = torch.nan_to_num(xyz)
	idx = torch.arange(sum(L_s))
	idx[L_s[0]:] += 200

	chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
	chain_idx[:L_s[0], :L_s[0]] = 1
	chain_idx[L_s[0]:, L_s[0]:] = 1

	# Do cropping
	if sum(L_s) > params['CROP']:
		if negative:
			sel = get_complex_crop(L_s, mask, msa.device, params)
		else:
			sel = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params, dom_pair)
		msa = msa[:,sel]
		xyz = xyz[sel]
		mask = mask[sel]
		xyz_t = xyz_t[:,sel]
		f1d_t = f1d_t[:,sel]
		mask_t = mask_t[:,sel]
		idx = idx[sel]
		chain_idx = chain_idx[sel][:,sel]
	return msa.long(),xyz_t.float(),f1d_t.float(),mask_t,xyz.float(),mask,idx.long(),chain_idx,False,negative


# filter the complex training dataset to make sure that they have the same taxID
def loader_complex(item, L_s, params, negative=False, random_noise=5.0):
	pdb_pair = item[0]
	pdbidA, pdbidB = item[0].split(':')
	hashA, hashB = item[1].split(':')
	a3m = {}
	if negative:
		pMSA_fn = params['COMPL_DIR'] + '/nega_pmsas/' + hashA[:3] + '/' + hashA + '_' + hashB + '.i95c50.a3m'
	else:
		pMSA_fn = params['COMPL_DIR'] + '/posi_pmsas/' + hashA[:3] + '/' + hashA + '_' + hashB + '.i95c50.a3m'
	a3m = get_msa(pMSA_fn, pdb_pair)
	msa = a3m['msa']
	if len(msa) > params['BLOCKCUT']:
		msa = MSABlockDeletion(msa)

	# generate empty template vectors
	xyz_t_A = INIT_CRDS.reshape(1,1,27,3).repeat(1,L_s[0],1,1) + torch.rand(1,L_s[0],1,3)*random_noise
	f1d_t_A = torch.nn.functional.one_hot(torch.full((1, L_s[0]), 20).long(), num_classes=21).float()
	conf_A = torch.zeros((1, L_s[0], 1)).float()
	f1d_t_A = torch.cat((f1d_t_A, conf_A), -1)
	mask_t_A = torch.full((1, L_s[0], 27), False)

	xyz_t_B = INIT_CRDS.reshape(1,1,27,3).repeat(1,L_s[1],1,1) + torch.rand(1,L_s[1],1,3)*random_noise
	f1d_t_B = torch.nn.functional.one_hot(torch.full((1, L_s[1]), 20).long(), num_classes=21).float()
	conf_B = torch.zeros((1, L_s[1], 1)).float()
	f1d_t_B = torch.cat((f1d_t_B, conf_B), -1)
	mask_t_B = torch.full((1, L_s[1], 27), False)

	xyz_t = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B)), dim=1) # (T, L1+L2, natm, 3)
	f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=1) # (T, L1+L2, natm, 3)
	mask_t = torch.cat((mask_t_A, mask_t_B), dim=1) # (T, L1+L2, natm, 3)

	# read PDB
	if negative:
		pdbA = get_pdb_miss(params['COMPL_DIR'] + '/nega_pdbs/' + pdbidA[1:3] + '/' + pdbidA + '-' + pdbidB + '__' + \
						pdbidA + '.pdb', L_s[0], pdbidA)
		pdbB = get_pdb_miss(params['COMPL_DIR'] + '/nega_pdbs/' + pdbidB[1:3] + '/' + pdbidA + '-' + pdbidB + '__' + \
						pdbidB + '.pdb', L_s[1], pdbidB)
	else:
		pdbA = get_pdb_miss(params['COMPL_DIR'] + '/posi_pdbs/' + pdbidA[1:3] + '/' + pdbidA + '-' + pdbidB + '__' + \
						pdbidA + '.pdb', L_s[0], pdbidA)
		pdbB = get_pdb_miss(params['COMPL_DIR'] + '/posi_pdbs/' + pdbidB[1:3] + '/' + pdbidA + '-' + pdbidB + '__' + \
						pdbidB + '.pdb', L_s[1], pdbidB)

	xyz = INIT_CRDS.reshape(1, 27, 3).repeat(sum(L_s), 1, 1)
	xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
	mask = torch.full((sum(L_s), 27), False)
	mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
	xyz = torch.nan_to_num(xyz)
	idx = torch.arange(sum(L_s))
	idx[L_s[0]:] += 200

	chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
	chain_idx[:L_s[0], :L_s[0]] = 1
	chain_idx[L_s[0]:, L_s[0]:] = 1

	# Do cropping
	if sum(L_s) > params['CROP']:
		if negative:
			sel = get_complex_crop(L_s, mask, msa.device, params)
		else:
			sel = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params, pdb_pair)
		msa = msa[:,sel]
		xyz = xyz[sel]
		mask = mask[sel]
		xyz_t = xyz_t[:,sel]
		f1d_t = f1d_t[:,sel]
		mask_t = mask_t[:,sel]
		idx = idx[sel]
		chain_idx = chain_idx[sel][:,sel]
	return msa.long(),xyz_t.float(),f1d_t.float(),mask_t,xyz.float(),mask,idx.long(),chain_idx,False,negative


class Dataset(data.Dataset):
	def __init__(self, IDs, loader, item_dict, params):
		self.IDs = IDs
		self.item_dict = item_dict
		self.loader = loader
		self.params = params
	def __len__(self):
		return len(self.IDs)
	def __getitem__(self, index):
		ID = self.IDs[index]
		sel_idx = np.random.randint(0, len(self.item_dict[ID]))
		out = self.loader(self.item_dict[ID][sel_idx][0], self.params)
		return out

class DatasetComplex(data.Dataset):
	def __init__(self, IDs, loader, item_dict, params, negative=False):
		self.IDs = IDs
		self.item_dict = item_dict
		self.loader = loader
		self.params = params
		self.negative = negative
	def __len__(self):
		return len(self.IDs)
	def __getitem__(self, index):
		ID = self.IDs[index]
		sel_idx = np.random.randint(0, len(self.item_dict[ID]))
		out = self.loader(self.item_dict[ID][sel_idx][0], self.item_dict[ID][sel_idx][1], self.params, \
						negative = self.negative)
		return out


class DistilledDataset(data.Dataset):
	def __init__(self, pdb_IDs, pdb_loader, pdb_dict, compl_IDs, compl_loader, compl_dict, \
				 neg_IDs, neg_loader, neg_dict, fb_IDs, fb_loader, fb_dict, muld_IDs, \
				 muld_loader, muld_dict, muld_neg_IDs, muld_neg_loader, muld_neg_dict, params):
		self.pdb_IDs = pdb_IDs
		self.pdb_loader = pdb_loader
		self.pdb_dict = pdb_dict
		self.compl_IDs = compl_IDs
		self.compl_loader = compl_loader
		self.compl_dict = compl_dict
		self.neg_IDs = neg_IDs
		self.neg_loader = neg_loader
		self.neg_dict = neg_dict
		self.fb_IDs = fb_IDs
		self.fb_loader = fb_loader
		self.fb_dict = fb_dict
		self.muld_IDs = muld_IDs
		self.muld_loader = muld_loader
		self.muld_dict = muld_dict
		self.muld_neg_IDs = muld_neg_IDs
		self.muld_neg_loader = muld_neg_loader
		self.muld_neg_dict = muld_neg_dict
		self.params = params
		
		self.pdb_inds = np.arange(len(self.pdb_IDs))
		self.compl_inds = np.arange(len(self.compl_IDs))
		self.neg_inds = np.arange(len(self.neg_IDs))
		self.fb_inds = np.arange(len(self.fb_IDs))
		self.muld_inds = np.arange(len(self.muld_IDs))
		self.muld_neg_inds = np.arange(len(self.muld_neg_IDs))
	
	def __len__(self):
		return len(self.pdb_inds) + len(self.compl_inds) + len(self.neg_inds) + len(self.fb_inds) \
						+ len(self.muld_inds) + len(self.muld_neg_inds)

	def __getitem__(self, index):
		if index >= len(self.pdb_inds) + len(self.compl_inds) + len(self.neg_inds) + len(self.fb_inds) \
						+ len(self.muld_inds):
			# from multidomain negative dataset
			ID = self.muld_neg_IDs[index - len(self.pdb_inds) - len(self.compl_inds) - len(self.neg_inds) \
							- len(self.fb_inds) - len(self.muld_inds)]
			sel_idx = np.random.randint(0, len(self.muld_neg_dict[ID]))
			out = self.muld_loader(self.muld_neg_dict[ID][sel_idx][0], self.muld_neg_dict[ID][sel_idx][1], \
							self.params, negative=True)

		elif index >= len(self.pdb_inds) + len(self.compl_inds) + len(self.neg_inds) + len(self.fb_inds):
			# from multidomain dataset
			ID = self.muld_IDs[index - len(self.pdb_inds) - len(self.compl_inds) - len(self.neg_inds) \
							- len(self.fb_inds)]
			sel_idx = np.random.randint(0, len(self.muld_dict[ID]))
			out = self.muld_loader(self.muld_dict[ID][sel_idx][0], self.muld_dict[ID][sel_idx][1], \
							self.params, negative=False)

		elif index >= len(self.pdb_inds) + len(self.compl_inds) + len(self.neg_inds): 
			# from FB set
			ID = self.fb_IDs[index - len(self.pdb_inds) - len(self.compl_inds) - len(self.neg_inds)]
			sel_idx = np.random.randint(0, len(self.fb_dict[ID]))
			out = self.fb_loader(self.fb_dict[ID][sel_idx][0], self.params)

		elif index >= len(self.pdb_inds) + len(self.compl_inds): 
			# from negative set
			ID = self.neg_IDs[index - len(self.pdb_inds) - len(self.compl_inds)]
			sel_idx = np.random.randint(0, len(self.neg_dict[ID]))
			out = self.neg_loader(self.neg_dict[ID][sel_idx][0], self.neg_dict[ID][sel_idx][1], \
							self.params, negative=True)

		elif index >= len(self.pdb_inds):
			# from complex set
			ID = self.compl_IDs[index - len(self.pdb_inds)]
			sel_idx = np.random.randint(0, len(self.compl_dict[ID]))
			out = self.compl_loader(self.compl_dict[ID][sel_idx][0], self.compl_dict[ID][sel_idx][1], \
							self.params, negative=False)

		else:
			# from PDB set
			ID = self.pdb_IDs[index]
			sel_idx = np.random.randint(0, len(self.pdb_dict[ID]))
			out = self.pdb_loader(self.pdb_dict[ID][sel_idx][0], self.params)		
		return out


class DistributedWeightedSampler(data.Sampler):
	def __init__(self, dataset, pdb_weights, compl_weights, neg_weights, fb_weights, \
					muld_weights, muld_neg_weights, num_example_per_epoch=25600, fraction_pdb=0.08, \
					fraction_compl=0.17, fraction_neg=0.17, fraction_fb=0.24, fraction_muld=0.17, \
					num_replicas=None, rank=None, replacement=False):
		if num_replicas is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			num_replicas = dist.get_world_size()
		if rank is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			rank = dist.get_rank()
		assert num_example_per_epoch % num_replicas == 0

		self.dataset = dataset
		self.num_replicas = num_replicas
		self.num_pdb_per_epoch = int(round(num_example_per_epoch * fraction_pdb))
		self.num_compl_per_epoch = int(round(num_example_per_epoch * fraction_compl))
		self.num_neg_per_epoch = int(round(num_example_per_epoch * fraction_neg))
		self.num_fb_per_epoch = int(round(num_example_per_epoch * fraction_fb))
		self.num_muld_per_epoch = int(round(num_example_per_epoch * fraction_muld))
		self.num_muld_neg_per_epoch = num_example_per_epoch - self.num_pdb_per_epoch \
						- self.num_compl_per_epoch - self.num_neg_per_epoch \
						- self.num_fb_per_epoch - self.num_muld_per_epoch
		print (self.num_pdb_per_epoch, self.num_compl_per_epoch, self.num_neg_per_epoch, \
						self.num_fb_per_epoch, self.num_muld_per_epoch, self.num_muld_neg_per_epoch)
		self.total_size = num_example_per_epoch
		self.num_samples = self.total_size // self.num_replicas
		self.rank = rank
		self.epoch = 0
		self.replacement = replacement
		self.pdb_weights = pdb_weights
		self.compl_weights = compl_weights
		self.neg_weights = neg_weights
		self.fb_weights = fb_weights
		self.muld_weights = muld_weights
		self.muld_neg_weights = muld_neg_weights

	def __iter__(self):
		# deterministically shuffle based on epoch
		g = torch.Generator()
		g.manual_seed(self.epoch)
		indices = torch.arange(len(self.dataset))
		# weighted subsampling
		sel_indices = torch.tensor((),dtype=int)

		if (self.num_pdb_per_epoch>0):
			pdb_sampled = torch.multinomial(self.pdb_weights, \
							self.num_pdb_per_epoch, self.replacement, generator=g)
			sel_indices = torch.cat((sel_indices, indices[pdb_sampled]))

		if (self.num_compl_per_epoch>0):
			compl_sampled = torch.multinomial(self.compl_weights, \
							self.num_compl_per_epoch, self.replacement, generator=g)
			sel_indices = torch.cat((sel_indices, indices[compl_sampled + len(self.dataset.pdb_IDs)]))
		
		if (self.num_neg_per_epoch>0):
			neg_sampled = torch.multinomial(self.neg_weights, \
							self.num_neg_per_epoch, self.replacement, generator=g)
			sel_indices = torch.cat((sel_indices, indices[neg_sampled + len(self.dataset.pdb_IDs) \
							+ len(self.dataset.compl_IDs)]))

		if (self.num_fb_per_epoch>0):
			fb_sampled = torch.multinomial(self.fb_weights, \
							self.num_fb_per_epoch, self.replacement, generator=g)
			sel_indices = torch.cat((sel_indices, indices[fb_sampled + len(self.dataset.pdb_IDs) \
							+ len(self.dataset.compl_IDs) + len(self.dataset.neg_IDs)]))

		if (self.num_muld_per_epoch>0):
			muld_sampled = torch.multinomial(self.muld_weights, \
							self.num_muld_per_epoch, self.replacement, generator=g)
			sel_indices = torch.cat((sel_indices, indices[muld_sampled + len(self.dataset.pdb_IDs) \
							+ len(self.dataset.compl_IDs) + len(self.dataset.neg_IDs) + len(self.dataset.fb_IDs)]))

		if (self.num_muld_neg_per_epoch>0):
			muld_neg_sampled = torch.multinomial(self.muld_neg_weights, \
							self.num_muld_neg_per_epoch, self.replacement, generator=g)
			sel_indices = torch.cat((sel_indices, indices[muld_neg_sampled + len(self.dataset.pdb_IDs) \
							+ len(self.dataset.compl_IDs) + len(self.dataset.neg_IDs) \
							+ len(self.dataset.fb_IDs) + len(self.dataset.muld_IDs)]))

		# shuffle indices
		indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]
		indices = indices[self.rank:self.total_size:self.num_replicas]
		assert len(indices) == self.num_samples
		return iter(indices.tolist())

	def __len__(self):
		return self.num_samples
	def set_epoch(self, epoch):
		self.epoch = epoch
