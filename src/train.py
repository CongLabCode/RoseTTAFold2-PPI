import sys, os, time, torch
from contextlib import ExitStack, nullcontext
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch.nn as nn
from torch.utils import data
from data_loader import get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_muld, MSAFeaturize, Dataset, DatasetComplex, DistilledDataset, DistributedWeightedSampler
from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, BIN_CONTACT
from RoseTTAFoldModel import RF2trackModule
from loss import *
from util import *
from scheduler import get_stepwise_decay_schedule_with_warmup

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#torch.autograd.set_detect_anomaly(True) # For NaN debugging

USE_AMP = False
N_PRINT_TRAIN = 32
# num structs per epoch
# must be divisible by #GPUs
N_EXAMPLE_PER_EPOCH = 2*6400

LOAD_PARAM = {'shuffle': False,
			  'num_workers': 4,
			  'pin_memory': True}


class EMA(nn.Module):
	def __init__(self, model, decay):
		super().__init__()
		self.decay = decay
		self.model = model
		self.shadow = deepcopy(self.model)
		for param in self.shadow.parameters():
			param.detach_()

	@torch.no_grad()
	def update(self):
		if not self.training:
			print("EMA update should only be called during training", file=stderr, flush=True)
			return
		model_params = OrderedDict(self.model.named_parameters())
		shadow_params = OrderedDict(self.shadow.named_parameters())
		# check if both model contains the same set of keys
		assert model_params.keys() == shadow_params.keys()

		for name, param in model_params.items():
			# see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
			# shadow_variable -= (1 - decay) * (shadow_variable - variable)
			shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))
		model_buffers = OrderedDict(self.model.named_buffers())
		shadow_buffers = OrderedDict(self.shadow.named_buffers())

		# check if both model contains the same set of keys
		assert model_buffers.keys() == shadow_buffers.keys()
		for name, buffer in model_buffers.items():
			# buffers are copied
			shadow_buffers[name].copy_(buffer)

	def forward(self, *args, **kwargs):
		if self.training:
			return self.model(*args, **kwargs)
		else:
			return self.shadow(*args, **kwargs)


class Trainer():
	def __init__(self, model_name='RF-PPI', n_epoch=100, lr=1.0e-4, port=None, interactive=False, \
					model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1, maxcycle=3):
		self.model_name = model_name
		self.n_epoch = n_epoch
		self.init_lr = lr
		self.port = port
		self.interactive = interactive
		self.model_param = model_param
		self.loader_param = loader_param
		self.loss_param = loss_param
		self.ACCUM_STEP = accum_step
		self.batch_size = batch_size
		# loss & final activation function
		self.loss_fn = nn.CrossEntropyLoss(reduction='none')
		self.active_fn = nn.Softmax(dim=1)
		self.maxcycle = maxcycle
		print (model_param, loader_param, loss_param)


	#def calc_loss(self, logit_s, label_s, logits_bind, mask_2d, same_chain,
	def calc_loss(self, logit_s, label_s, p_bind, mask_2d, same_chain, logit_aa_s, label_aa_s, \
					mask_aa_s, logit_exp, mask_BB, eval_PPI=False, negative=False, w_dist=1.0, \
					w_aa=1.0, w_exp=1.0, w_bind=0.0, eps=1e-6):
		'''
		Inputs:
		 - logit_s: 2D logits for distogram & orientogram [(B, C, L, L)*4]
		 - label_s: Labels for distogram & orientogram [(B, L, L)*4]
		 - mask_2d: 1 means Valid residue pairs (not including missing residues), (B, L, L)
		 - same_chain: whether two residues in the same chain (1 - same chain / 0 - different chain), (B, L, L)
		 - logit_aa_s: Logits for masked language modeling (MLM) task (B, C, N, L)
		 - label_aa_s: true labels for MLM task (B, N, L)
		 - mask_aa_s: masked positions for MLM task (1: masked, 0: not masked) (B, N, L)
		 - logit_exp: logits for predicting experimentally resolved region (B, L)
		 - mask_BB: whether they are experimentally resolved (1-resolved, 0-missing) (B, L)
		 - logits_bind: Logits for predicted probability to bind (B)
		 - eval_PPI: evaluate logits_bind or not
		 - negative: Whether given PPI is negative pairs or not
		 - w_*: weights for each loss terms

		Outputs:
		 - Loss of the given predictions
		'''
		B, L = mask_2d.shape[:2]
		loss_s = list()
		tot_loss = 0.0

		# col 0~3: c6d loss (distogram, orientogram prediction)
		# for negatives, labels for inter-chain regions should assign as "FAR APART" (the last bin) 
		loss = calc_c6d_loss(logit_s, label_s, mask_2d)
		tot_loss += w_dist * loss.sum()
		loss_s.append(loss.detach())

		# col 4: masked token prediction loss
		loss = self.loss_fn(logit_aa_s, label_aa_s)
		loss = loss * mask_aa_s
		loss = loss.sum() / (mask_aa_s.sum() + eps)
		tot_loss += w_aa * loss
		loss_s.append(loss[None].detach())

		# col 5: experimentally resolved prediction loss
		loss = nn.BCEWithLogitsLoss()(logit_exp, mask_BB.float())
		tot_loss += w_exp * loss
		loss_s.append(loss[None].detach())

		# col 6: binder loss
		if eval_PPI:
			if (negative):
				target = torch.tensor([0.0], device=p_bind.device)
			else:
				target = torch.tensor([1.0], device=p_bind.device)
			loss = torch.nn.BCELoss()(p_bind,target)
			#loss = torch.nn.BCEWithLogitsLoss()(logits_bind,target)
			tot_loss += w_bind * loss
		else:
			loss = torch.tensor(0.0, device=logit_s[0].device)
		loss_s.append(loss[None].detach())
		return tot_loss, torch.cat(loss_s, dim=0)


	def load_model(self, model, optimizer, scheduler, scaler, model_name, rank, suffix='last', resume_train=True):
		chk_fn = "models/%s_%s.pt"%(model_name, suffix)
		loaded_epoch = -1
		best_valid_F1 = -9999.9
		if not os.path.exists(chk_fn):
			print ('no model found', model_name)
			return -1, best_valid_F1
		map_location = {"cuda:%d"%0: "cuda:%d"%rank}
		checkpoint = torch.load(chk_fn, map_location=map_location)

		model.module.model.load_state_dict(checkpoint['final_state_dict'], strict=True)
		model.module.shadow.load_state_dict(checkpoint['model_state_dict'], strict=True)
		try:
			best_valid_F1 = checkpoint['best_F1_50']
		except KeyError:
			best_valid_F1 = 0.5
		if resume_train:
			loaded_epoch = checkpoint['epoch']
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			scaler.load_state_dict(checkpoint['scaler_state_dict'])
			if 'scheduler_state_dict' in checkpoint:
				scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			else:
				scheduler.last_epoch = loaded_epoch + 1
		return loaded_epoch, best_valid_F1


	def checkpoint_fn(self, model_name, description):
		if not os.path.exists("models"):
			os.mkdir("models")
		name = "%s_%s.pt"%(model_name, description)
		return os.path.join("models", name)


	# main entry function of training
	# 1) make sure ddp env vars set
	# 2) figure out if we launched using slurm or interactively
	#	- if slurm, assume 1 job launched per GPU
	#	- if interactive, launch one job for each GPU on node
	def run_model_training(self, world_size):
		if ('MASTER_ADDR' not in os.environ):
			os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
		if ('MASTER_PORT' not in os.environ):
			os.environ['MASTER_PORT'] = '%d'%self.port
		if (not self.interactive and "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ):
			world_size = int(os.environ["SLURM_NTASKS"])
			rank = int (os.environ["SLURM_PROCID"])
			print ("Launched from slurm", rank, world_size)
			self.train_model(rank, world_size)
		else:
			print ("Launched from interactive")
			world_size = torch.cuda.device_count()
			mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)


	def train_model(self, rank, world_size):
		print ("running ddp on rank %d, world_size %d"%(rank, world_size))
		gpu = rank % torch.cuda.device_count()
		dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
		torch.cuda.set_device("cuda:%d"%gpu)

		#define dataset & data loader
		pdb_items, fb_items, muld_items, muld_neg_items, compl_items, neg_items, \
						valid_pdb, valid_compl, valid_neg = get_train_valid_set(self.loader_param)
		pdb_IDs, pdb_weights, pdb_dict = pdb_items
		fb_IDs, fb_weights, fb_dict = fb_items
		muld_IDs, muld_weights, muld_dict = muld_items
		muld_neg_IDs, muld_neg_weights, muld_neg_dict = muld_neg_items
		compl_IDs, compl_weights, compl_dict = compl_items
		neg_IDs, neg_weights, neg_dict = neg_items

		self.n_train = N_EXAMPLE_PER_EPOCH
		self.n_valid_pdb = min(480, len(valid_pdb.keys()))
		self.n_valid_pdb = (self.n_valid_pdb // world_size) * world_size
		self.n_valid_compl = min(480, len(valid_compl.keys()))
		self.n_valid_compl = (self.n_valid_compl // world_size) * world_size
		self.n_valid_neg = min(480, len(valid_neg.keys()))
		self.n_valid_neg = (self.n_valid_neg // world_size) * world_size

		train_set = DistilledDataset(pdb_IDs, loader_pdb, pdb_dict, compl_IDs, loader_complex, compl_dict, \
						neg_IDs, loader_complex, neg_dict, fb_IDs, loader_fb, fb_dict, muld_IDs, loader_muld, \
						muld_dict, muld_neg_IDs, loader_muld, muld_neg_dict, self.loader_param)		
		valid_pdb_set = Dataset(list(valid_pdb.keys())[:self.n_valid_pdb], loader_pdb, \
						valid_pdb, self.loader_param)
		valid_compl_set = DatasetComplex(list(valid_compl.keys())[:self.n_valid_compl], \
						loader_complex, valid_compl, self.loader_param, negative=False)
		valid_neg_set = DatasetComplex(list(valid_neg.keys())[:self.n_valid_neg], loader_complex, \
						valid_neg, self.loader_param, negative=True)

		train_sampler = DistributedWeightedSampler(train_set, pdb_weights, compl_weights, neg_weights, \
						fb_weights, muld_weights, muld_neg_weights, num_example_per_epoch=N_EXAMPLE_PER_EPOCH, \
						num_replicas=world_size, rank=rank, fraction_pdb=0.08, fraction_compl=0.17, \
						fraction_neg=0.17, fraction_fb=0.24, fraction_muld=0.17)
		valid_pdb_sampler = data.distributed.DistributedSampler(valid_pdb_set, num_replicas=world_size, rank=rank)
		valid_compl_sampler = data.distributed.DistributedSampler(valid_compl_set, num_replicas=world_size, rank=rank)
		valid_neg_sampler = data.distributed.DistributedSampler(valid_neg_set, num_replicas=world_size, rank=rank)

		train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, **LOAD_PARAM)
		valid_pdb_loader = data.DataLoader(valid_pdb_set, sampler=valid_pdb_sampler, **LOAD_PARAM)
		valid_compl_loader = data.DataLoader(valid_compl_set, sampler=valid_compl_sampler, **LOAD_PARAM)
		valid_neg_loader = data.DataLoader(valid_neg_set, sampler=valid_neg_sampler, **LOAD_PARAM)

		# define model
		model = EMA(RF2trackModule(**self.model_param).to(gpu), 0.99)
		ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
		# ddp_model._set_static_graph() # required to use gradient checkpointing w/ shared parameters
		if rank == 0:
			print ("# of parameters:", sum(p.numel() for p in ddp_model.parameters() if p.requires_grad))

		# define optimizer and scheduler
		optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=self.init_lr)
		#scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 1000, 15000, 0.95)
		scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 10000, 0.95) # Use this for fine-tuning 
		scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP) 
		# make resume_train=False when you start fine-tuning
		loaded_epoch, best_valid_F1 = self.load_model(ddp_model, optimizer, scheduler, scaler, \
						self.model_name, gpu, resume_train=True)

		if loaded_epoch >= self.n_epoch:
			DDP_cleanup()
			return

		for epoch in range(loaded_epoch+1, self.n_epoch):
			train_sampler.set_epoch(epoch)
			valid_pdb_sampler.set_epoch(epoch)
			valid_compl_sampler.set_epoch(epoch)
			valid_neg_sampler.set_epoch(epoch)
			train_tot, train_loss = self.train_cycle(ddp_model, train_loader, optimizer, \
							scheduler, scaler, rank, gpu, world_size, epoch)
			valid_tot, valid_loss = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, epoch)
			F1_50 = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, epoch)

			if rank == 0: # save model
				if (F1_50 > best_valid_F1):
					best_valid_F1 = F1_50
					torch.save({'epoch': epoch,
								'model_state_dict': ddp_model.module.shadow.state_dict(),
								'final_state_dict': ddp_model.module.model.state_dict(),
								'optimizer_state_dict': optimizer.state_dict(),
								'scheduler_state_dict': scheduler.state_dict(),
								'scaler_state_dict': scaler.state_dict(),
								'best_F1_50': best_valid_F1,
								'train_loss': train_loss,
								'valid_loss': valid_loss},
								self.checkpoint_fn(self.model_name, 'best'))

				# Save the latest model & every 10-th checkpoints 
				torch.save({'epoch': epoch,
							'model_state_dict': ddp_model.module.shadow.state_dict(),
							'final_state_dict': ddp_model.module.model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'scheduler_state_dict': scheduler.state_dict(),
							'scaler_state_dict': scaler.state_dict(),
							'train_loss': train_loss,
							'best_F1_50': best_valid_F1},
							self.checkpoint_fn(self.model_name, 'last'))

				if (epoch % 10 == 0):
					torch.save({'epoch': epoch,
								'model_state_dict': ddp_model.module.shadow.state_dict(),
								'final_state_dict': ddp_model.module.model.state_dict(),
								'optimizer_state_dict': optimizer.state_dict(),
								'scheduler_state_dict': scheduler.state_dict(),
								'scaler_state_dict': scaler.state_dict(),
								'train_loss': train_loss,
								'valid_loss': valid_loss,
								'F1_50': F1_50,
								'best_F1_50': best_valid_F1},
								self.checkpoint_fn(self.model_name, 'epoch%03d'%epoch))
		dist.destroy_process_group()


	def _prepare_input_common(self, inputs, gpu):
		'''
		Prepare common inputs that used in all recycling step
		This function will copy arrays to device (gpu) & prepare template 2D features from given xyz_t
		Featurization of MSA inputs will happen separately because it will have different sub-sampled MSA at each recycle
		Arrays in inputs
		 - msa_orig: original MSA for cropped region having shape of (Batch, N_seq, Length)
		 - xyz_t: xyz coordinates of templates (B, T, L, N_atom, 3)
		 - t1d: 1-D features of templates (one-hot encoded template seq + 1D confidence) (B, T, L, d_t1d)
		 - mask_t: valid atoms in templates (1: valid / 0: missing) (B, T, L, N_atom)
		 - true_crds: ground-truth coordinates of target protein (label) (B, L, N_atom, 3)
		 - mask_crds: valid atoms in ground-truth structures (B, L, N_atom)
		 - idx_pdb: residue indices in original sequence (B, L)
		 - same_chain: whether two residues are in the same chain or not (B, L, L)
		 - eval_PPI: evaluate p_bind or not
		 - negative: True (negative PPI) or False (true PPI) 
		Return
		 - network_input (msa_orig, idx, t1d, t2d, same_chain)
		 - labels & other required info to calculate loss: true_crds, mask_crds, eval_PPI, negative
		'''
		msa_orig, xyz_t, t1d, mask_t, true_crds, mask_crds, idx_pdb, same_chain, eval_PPI, negative = inputs
		
		# 1. transfer inputs to device
		msa_orig = msa_orig.to(gpu, non_blocking=True)
		xyz_t = xyz_t.to(gpu, non_blocking=True)
		t1d = t1d.to(gpu, non_blocking=True)
		mask_t = mask_t.to(gpu, non_blocking=True)
		true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
		mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
		idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
		same_chain = same_chain.to(gpu, non_blocking=True)

		# 2. processing template features
		mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
		mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
		mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None]
		# (ignore inter-chain region because we don't use any complex templates)
		t2d = xyz_to_t2d(xyz_t, mask_t_2d) # 2D features for templates (B, T, L, L, 37+6+1)

		network_input = {}
		network_input['msa_orig'] = msa_orig
		network_input['idx'] = idx_pdb
		network_input['t1d'] = t1d
		network_input['t2d'] = t2d
		network_input['same_chain'] = same_chain
		return network_input, true_crds, mask_crds, eval_PPI, negative


	def _get_model_input(self, network_input, output_i, return_raw=False, use_checkpoint=False):
		'''
		Get network inputs for the current recycling step
		It will return dictionary of network inputs & labels for MLM tasks (mask_msa & msa_seed_gt)
		'''
		input_i = {}
		for key in network_input:
			if key == 'msa_orig':
				seq, msa_seed, msa_extra, mask_msa, msa_seed_gt = \
								MSAFeaturize(network_input['msa_orig'], self.loader_param)
				input_i['seq'] = seq # masked query sequence
				input_i['msa_seed'] = msa_seed # masked seed MSA
				input_i['msa_extra'] = msa_extra # extra MSA
			else:
				input_i[key] = network_input[key]
		msa_prev, pair_prev = output_i
		input_i['msa_prev'] = msa_prev
		input_i['pair_prev'] = pair_prev
		input_i['return_raw'] = return_raw
		input_i['use_checkpoint'] = use_checkpoint
		return input_i, mask_msa, msa_seed_gt


	def _get_loss_and_misc(self, output_i, true_msa, mask_msa, true_crds, mask_crds, \
					same_chain, eval_PPI=False, negative=False, return_bind=False):
		logit_c6d_s, logit_aa_s, logit_exp = output_i
		# processing labels for distogram orientograms
		mask_BB = ~(mask_crds[:,:,:3].sum(dim=-1) < 3.0) # ignore residues having missing BB atoms for loss calculation
		mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
		label_c6d = xyz_to_c6d(true_crds)
		label_c6d = c6d_to_bins2(label_c6d, same_chain, negative=negative)

		# calculate p_bind from predicted distogram
		p_bind = self.active_fn(logit_c6d_s[0])[:,:BIN_CONTACT].sum(dim=1)
		p_bind = p_bind*(1.0-same_chain.float()) # (B, L, L)
		p_bind = nn.MaxPool2d(p_bind.shape[1:])(p_bind).view(-1)
		p_bind = torch.clamp(p_bind, min=0.0, max=1.0)
		#logits_bind = torch.log(p_bind) - torch.log(1-p_bind)

		#loss, loss_s = self.calc_loss(logit_c6d_s, label_c6d, logits_bind, mask_2d, same_chain,
		loss, loss_s = self.calc_loss(logit_c6d_s, label_c6d, p_bind, mask_2d, same_chain, logit_aa_s, true_msa, \
						mask_msa, logit_exp, mask_BB, eval_PPI=eval_PPI, negative=negative, **self.loss_param)
		if return_bind:
			return loss, loss_s, p_bind
		else:
			return loss, loss_s


	def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch):
		# Turn on training mode
		ddp_model.train()
		# clear gradients
		optimizer.zero_grad()
		start_time = time.time()
		
		# For intermediate logs
		local_tot = 0.0
		local_loss = None
		train_tot = 0.0
		train_loss = None
		counter = 0

		for inputs in train_loader:
			network_input, true_crds, mask_crds, eval_PPI, negative = self._prepare_input_common(inputs, gpu)
			counter += 1
			N_cycle = np.random.randint(1, self.maxcycle+1) # number of recycling

			output_i = (None, None) # (query_seq_prev, pair_prev)
			for i_cycle in range(N_cycle):
				with ExitStack() as stack:
					if i_cycle < N_cycle -1:
						stack.enter_context(torch.no_grad())
						stack.enter_context(ddp_model.no_sync())
						stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
						return_raw=True
						use_checkpoint=False
					else:
						stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
						return_raw=False
						use_checkpoint=False
					
					input_i, mask_msa, msa_seed_gt = self._get_model_input(network_input, output_i, \
									return_raw=return_raw, use_checkpoint=use_checkpoint)
					output_i = ddp_model(**input_i)
					if i_cycle < N_cycle - 1:
						continue
					loss, loss_s = self._get_loss_and_misc(output_i, msa_seed_gt, mask_msa, true_crds, \
									mask_crds, network_input['same_chain'], eval_PPI=eval_PPI, negative=negative)

			loss = loss / self.ACCUM_STEP
			scaler.scale(loss).backward()
			if counter%self.ACCUM_STEP == 0:  
				# gradient clipping
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
				scaler.step(optimizer)
				scale = scaler.get_scale()
				scaler.update()
				skip_lr_sched = (scale != scaler.get_scale())
				optimizer.zero_grad()
				if not skip_lr_sched:
					scheduler.step()
				ddp_model.module.update() # apply EMA

			local_tot += loss.detach()*self.ACCUM_STEP
			if local_loss == None:
				local_loss = torch.zeros_like(loss_s.detach())
			local_loss += loss_s.detach()

			train_tot += loss.detach()*self.ACCUM_STEP
			if train_loss == None:
				train_loss = torch.zeros_like(loss_s.detach())
			train_loss += loss_s.detach()

			if counter % N_PRINT_TRAIN == 0:
				if rank == 0:
					max_mem = torch.cuda.max_memory_allocated()/1e9
					train_time = time.time() - start_time
					local_tot /= float(N_PRINT_TRAIN)
					local_loss /= float(N_PRINT_TRAIN)
					local_tot = local_tot.cpu().detach()
					local_loss = local_loss.cpu().detach().numpy()
					local_loss_str = " ".join(["%8.4f"%l for l in local_loss])
					sys.stdout.write("Local: [%04d/%04d] Batch: [%05d/%05d] Time: %8.1f | Loss: %8.4f | %s | Max mem %.4f\n" \
									% (epoch, self.n_epoch, counter*self.batch_size*world_size, \
									self.n_train, train_time, local_tot, local_loss_str, max_mem))
					sys.stdout.flush()
					local_tot = 0.0
					local_loss = None 
				torch.cuda.reset_peak_memory_stats()
			torch.cuda.empty_cache()

		# write total train loss
		train_tot /= float(counter * world_size)
		train_loss /= float(counter * world_size)
		dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
		dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
		train_tot = train_tot.cpu().detach()
		train_loss = train_loss.cpu().detach().numpy()
		if rank == 0:
			train_time = time.time() - start_time
			sys.stdout.write("Train: [%04d/%04d] Time: %8.1f | Loss: %8.4f | %s\n"%(epoch, self.n_epoch, \
							train_time, train_tot, " ".join(["%8.4f"%l for l in train_loss])))
			sys.stdout.flush()
		return train_tot, train_loss


	def valid_pdb_cycle(self, ddp_model, valid_loader, rank, gpu, world_size, epoch):
		valid_tot = 0.0
		valid_loss = None
		counter = 0
		start_time = time.time()

		with torch.no_grad(): # no need to calculate gradient
			ddp_model.eval() # change it to eval mode
			for inputs in valid_loader:
				network_input, true_crds, mask_crds, eval_PPI, negative = self._prepare_input_common(inputs, gpu)
				counter += 1
				N_cycle = self.maxcycle # number of recycling

				output_i = (None, None) # (msa_prev, pair_prev)
				for i_cycle in range(N_cycle):
					with ExitStack() as stack:
						stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
						stack.enter_context(ddp_model.no_sync())
						use_checkpoint=False
						if i_cycle < N_cycle -1:
							return_raw=True
						else:
							return_raw=False
						input_i, mask_msa, msa_seed_gt = self._get_model_input(network_input, output_i, \
										return_raw=return_raw, use_checkpoint=use_checkpoint)
						output_i = ddp_model(**input_i)
						if i_cycle < N_cycle - 1:
							continue						
						loss, loss_s = self._get_loss_and_misc(output_i, msa_seed_gt, mask_msa, true_crds, \
										mask_crds, network_input['same_chain'], eval_PPI=eval_PPI, negative=negative)

				valid_tot += loss.detach()
				if valid_loss == None:
					valid_loss = torch.zeros_like(loss_s.detach())
				valid_loss += loss_s.detach()

		valid_tot /= float(counter*world_size)
		valid_loss /= float(counter*world_size)
		dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
		dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
		valid_tot = valid_tot.cpu().detach().numpy()
		valid_loss = valid_loss.cpu().detach().numpy()

		if rank == 0:
			train_time = time.time() - start_time
			sys.stdout.write("Monomer: [%04d/%04d] Time: %8.1f | Loss: %8.4f | %s\n"%(epoch, self.n_epoch, \
							train_time, valid_tot, " ".join(["%8.4f"%l for l in valid_loss])))
			sys.stdout.flush()
		return valid_tot, valid_loss


	def valid_ppi_cycle(self, ddp_model, valid_pos_loader, valid_neg_loader, rank, gpu, world_size, epoch):
		valid_tot = 0.0
		valid_loss = None
		counter = 0
		TP_50 = 0
		FP_50 = 0
		FN_50 = 0
		F1_50 = 0
		start_time = time.time()

		with torch.no_grad(): # no need to calculate gradient
			ddp_model.eval() # change it to eval mode
			for inputs in valid_pos_loader:
				network_input, true_crds, mask_crds, eval_PPI, negative = self._prepare_input_common(inputs, gpu)
				counter += 1
				N_cycle = self.maxcycle # number of recycling
				output_i = (None, None) # (msa_prev, pair_prev)

				for i_cycle in range(N_cycle): 
					with ExitStack() as stack:
						stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
						stack.enter_context(ddp_model.no_sync())
						use_checkpoint=False
						if i_cycle < N_cycle - 1:
							return_raw=True
						else:
							return_raw=False
						
						input_i, mask_msa, msa_seed_gt = self._get_model_input(network_input, \
										output_i, return_raw=return_raw, use_checkpoint=use_checkpoint)
						output_i = ddp_model(**input_i)
						if i_cycle < N_cycle-1:
							continue
						loss, loss_s, p_bind = self._get_loss_and_misc(output_i, msa_seed_gt, mask_msa, \
										true_crds, mask_crds, network_input['same_chain'], eval_PPI=eval_PPI, \
										negative=negative, return_bind=True)
						if p_bind > 0.5:
							TP_50 += 1.0
						else:
							FN_50 += 1.0

				valid_tot += loss.detach()
				if valid_loss == None:
					valid_loss = torch.zeros_like(loss_s.detach())
				valid_loss += loss_s.detach()
		valid_tot /= float(counter*world_size)
		valid_loss /= float(counter*world_size)
		dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
		dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
		valid_tot = valid_tot.cpu().detach().numpy()
		valid_loss = valid_loss.cpu().detach().numpy()

		if rank == 0:
			train_time = time.time() - start_time
			sys.stdout.write("Hetero: [%04d/%04d] Time: %8.1f | Loss: %8.4f | %s\n"%(epoch, self.n_epoch, \
							train_time, valid_tot, " ".join(["%8.4f"%l for l in valid_loss])))
			sys.stdout.flush()


		valid_tot = 0.0
		valid_loss = None
		counter = 0
		start_time = time.time()

		with torch.no_grad(): # no need to calculate gradient
			ddp_model.eval() # change it to eval mode
			for inputs in valid_neg_loader: 
				network_input, true_crds, mask_crds, eval_PPI, negative = self._prepare_input_common(inputs, gpu)
				counter += 1
				N_cycle = self.maxcycle # number of recycling

				output_i = (None, None) # (msa_prev, pair_prev)
				for i_cycle in range(N_cycle): 
					with ExitStack() as stack:
						stack.enter_context(torch.cuda.amp.autocast(enabled=USE_AMP))
						stack.enter_context(ddp_model.no_sync())
						if i_cycle < N_cycle - 1:
							return_raw=True
						else:
							return_raw=False

						input_i, mask_msa, msa_seed_gt = self._get_model_input(network_input, output_i, \
										return_raw=return_raw, use_checkpoint=use_checkpoint)
						output_i = ddp_model(**input_i)
						if i_cycle < N_cycle-1:
							continue
						loss, loss_s, p_bind = self._get_loss_and_misc(output_i, msa_seed_gt, mask_msa, \
										true_crds, mask_crds, network_input['same_chain'], eval_PPI=eval_PPI, \
										negative=negative, return_bind=True)
						if p_bind > 0.5:
							FP_50 += 1.0

				valid_tot += loss.detach()
				if valid_loss == None:
					valid_loss = torch.zeros_like(loss_s.detach())
				valid_loss += loss_s.detach()
		valid_inter_50 = torch.tensor([TP_50, FP_50, FN_50], device=valid_loss.device).float()
		valid_tot /= float(counter*world_size)
		valid_loss /= float(counter*world_size)

		dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
		dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
		dist.all_reduce(valid_inter_50, op=dist.ReduceOp.SUM)
		valid_tot = valid_tot.cpu().detach().numpy()
		valid_loss = valid_loss.cpu().detach().numpy()
		valid_inter_50 = valid_inter_50.cpu().detach().numpy()

		if rank == 0:
			TP_50, FP_50, FN_50 = valid_inter_50 
			prec_50 = TP_50/(TP_50+FP_50+1e-4)
			recall_50 = TP_50/(TP_50+FN_50+1e-4)
			F1_50 = 2*TP_50/(2*TP_50+FP_50+FN_50+1e-4)
			train_time = time.time() - start_time
			sys.stdout.write("PPI: [%04d/%04d] Time: %8.1f | Loss: %8.4f | %s | %.4f %.4f %.4f\n"%(epoch, \
							self.n_epoch, train_time, valid_tot, " ".join(["%8.4f"%l for l in valid_loss]),\
							prec_50, recall_50, F1_50))
			sys.stdout.flush()
		return F1_50


if __name__ == "__main__":
	from arguments import get_args
	args, model_param, loader_param, loss_param = get_args()
	# set random seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	mp.freeze_support()
	train = Trainer(model_name=args.model_name, interactive=args.interactive, n_epoch=args.num_epochs, \
					lr=args.lr, port=args.port, model_param=model_param, loader_param=loader_param, \
					loss_param=loss_param, batch_size=args.batch_size, accum_step=args.accum, maxcycle=args.maxcycle)
	train.run_model_training(torch.cuda.device_count())
