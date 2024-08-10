import argparse
import data_loader
import os

TRUNK_PARAMS = ['n_extra_block', 'n_main_block',\
				'd_msa', 'd_msa_extra', 'd_pair', 'd_templ',\
				'n_head_msa', 'n_head_pair', 'n_head_templ', 'd_hidden', 'd_hidden_templ', 'p_drop']

def get_args():
	parser = argparse.ArgumentParser()

	# training parameters
	train_group = parser.add_argument_group("training parameters")
	train_group.add_argument("-model_name", default="RF-PPI",
			help="model name for saving")
	train_group.add_argument('-batch_size', type=int, default=1,
			help="Batch size [1]")
	train_group.add_argument('-lr', type=float, default=1.0e-4, 
			help="Learning rate [1.0e-4]")
	train_group.add_argument('-num_epochs', type=int, default=141,
			help="Number of epochs [141]")
	train_group.add_argument("-port", type=int, default=12319,
			help="PORT for ddp training, should be randomized [12319]")
	train_group.add_argument("-seed", type=int, default=0,
			help="seed for random number, should be randomized for different training run [0]")
	train_group.add_argument("-accum", type=int, default=1,
			help="Gradient accumulation when it's > 1 [1]")
	train_group.add_argument("-interactive", action="store_true", default=False,
			help="Use interactive node")

	# data-loading parameters
	data_group = parser.add_argument_group("data loading parameters")
	data_group.add_argument('-maxseq', type=int, default=1024,
			help="Maximum depth of subsampled MSA [1024]")
	data_group.add_argument('-maxlat', type=int, default=128,
			help="Maximum depth of subsampled MSA [128]")
	data_group.add_argument("-crop", type=int, default=256,
			help="Upper limit of crop size [256]")
	data_group.add_argument('-mintplt', type=int, default=0,
			help="Minimum number of templates to select [0]")
	data_group.add_argument('-maxtplt', type=int, default=0,
			help="maximum number of templates to select [0]")
	data_group.add_argument("-rescut", type=float, default=5.0,
			help="Resolution cutoff [5.0]")
	data_group.add_argument("-datcut", default="2020-Apr-30",
			help="PDB release date cutoff [2020-Apr-30]")
	data_group.add_argument('-plddtcut', type=float, default=70.0,
			help="pLDDT cutoff for distillation set [70.0]")
	data_group.add_argument('-seqid', type=float, default=95.0,
			help="maximum sequence identity cutoff for template selection [95.0]")
	data_group.add_argument('-maxcycle', type=int, default=3,
			help="maximum number of recycle [3]")

	# Trunk module properties
	trunk_group = parser.add_argument_group("Trunk module parameters")
	trunk_group.add_argument('-n_extra_block', type=int, default=4,
			help="Number of iteration blocks for extra sequences [4]")
	trunk_group.add_argument('-n_main_block', type=int, default=12,
			help="Number of iteration blocks for main sequences [12]")
	trunk_group.add_argument('-d_msa', type=int, default=256,
			help="Number of MSA features [256]")
	trunk_group.add_argument('-d_msa_extra', type=int, default=64,
			help="Number of MSA features [64]")
	trunk_group.add_argument('-d_pair', type=int, default=128,
			help="Number of pair features [128]")
	trunk_group.add_argument('-d_templ', type=int, default=64,
			help="Number of templ features [64]")
	trunk_group.add_argument('-n_head_msa', type=int, default=8,
			help="Number of attention heads for MSA2MSA [8]")
	trunk_group.add_argument('-n_head_pair', type=int, default=4,
			help="Number of attention heads for Pair2Pair [4]")
	trunk_group.add_argument('-n_head_templ', type=int, default=4,
			help="Number of attention heads for template [4]")
	trunk_group.add_argument("-d_hidden", type=int, default=32,
			help="Number of hidden features [32]")
	trunk_group.add_argument("-d_hidden_templ", type=int, default=32,
			help="Number of hidden features for templates [32]")
	trunk_group.add_argument("-p_drop", type=float, default=0.15,
			help="Dropout ratio [0.15]")

	# Loss function parameters
	loss_group = parser.add_argument_group("loss parameters")
	loss_group.add_argument('-w_dist', type=float, default=2.0,
			help="Weight on distd in loss function [2.0]")
	loss_group.add_argument('-w_exp', type=float, default=0.1,
			help="Weight on experimental resolved in loss function [0.1]")
	loss_group.add_argument('-w_aa', type=float, default=1.0,
			help="Weight on MSA masked token prediction loss [1.0]")
	loss_group.add_argument('-w_bind', type=float, default=0.0,
			help="Weight on MSA masked token prediction loss [0.0]")

	# parse arguments
	args = parser.parse_args()

	# Setup dataloader parameters:
	loader_param = data_loader.set_data_loader_params(args)

	# make dictionary for each parameters
	trunk_param = {}
	for param in TRUNK_PARAMS:
		trunk_param[param] = getattr(args, param)
	loss_param = {}
	for param in ['w_dist', 'w_aa', 'w_exp', 'w_bind']:
		loss_param[param] = getattr(args, param)

	return args, trunk_param, loader_param, loss_param
