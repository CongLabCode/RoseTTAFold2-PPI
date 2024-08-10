import numpy as np
import scipy
import scipy.spatial
import string
import os,re
import random
import util
import gzip
import torch
from chemical import INIT_CRDS

to1letter = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E', \
				'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', \
				'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V' }

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename, max_seq=10000):
	msa = []
	table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
	if filename.split('.')[-1] == 'gz':
		fp = gzip.open(filename, 'rt')
	else:
		fp = open(filename, 'r')

	# read file line by line
	for line in fp:
		# skip labels
		if line[0] == '>':
			continue
		# remove right whitespaces
		line = line.rstrip()
		if len(line) == 0:
			continue
		# remove lowercase letters and append to MSA
		msa_i = line.translate(table)
		msa.append(msa_i)
		if len(msa) >= max_seq:
			break

	msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8) # (Nseq, L)
	# convert letters into numbers
	alphabet = np.array(list('ARNDCQEGHILKMFPSTWYVX-'), dtype='|S1').view(np.uint8)
	for i in range(alphabet.shape[0]):
		msa[msa == alphabet[i]] = i
	# treat all unknown characters as gaps
	msa[msa > 21] = 21
	return msa


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file
def parse_pdb(filename):
	lines = open(filename,'r').readlines()
	idx_s = [int(l[22:26]) for l in lines if l[:4]=='ATOM' and l[12:16].strip()=='CA']

	# 4 BB + up to 10 SC atoms
	xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
	for l in lines:
		if l[:4] != 'ATOM':
			continue
		resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
		idx = idx_s.index(resNo)
		for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
			if i_atm < 14:
				if tgtatm == atom:
					xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
					break
	
	mask = np.logical_not(np.isnan(xyz[...,0]))
	xyz[np.isnan(xyz[...,0])] = 0.0
	return xyz, mask, np.array(idx_s)


def parse_pdb_w_miss(filename, L):
	lines = open(filename,'r').readlines()
	idx_s = [int(l[22:26]) for l in lines if l[:4]=='ATOM' and l[12:16].strip()=='CA']

	xyz = np.full((L, 14, 3), np.nan, dtype=np.float32)
	for l in lines:
		if l[:4] != 'ATOM':
			continue
		resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
		for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
			if i_atm < 14:
				if tgtatm == atom:
					xyz[resNo-1,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
					break

	mask = np.logical_not(np.isnan(xyz[...,0]))
	xyz[np.isnan(xyz[...,0])] = 0.0
	return xyz, mask, np.array(idx_s)


def parse_pdb_w_plddt(filename):
	lines = open(filename,'r').readlines()
	idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
	lddts = [float(l[60:66]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]

	xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
	for l in lines:
		if l[:4] != "ATOM":
			continue
		resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
		idx = idx_s.index(resNo)
		for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
			if i_atm < 14:
				if tgtatm == atom:
					xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
					break

	mask = np.logical_not(np.isnan(xyz[...,0]))
	xyz[np.isnan(xyz[...,0])] = 0.0
	return xyz, mask, np.array(idx_s), np.array(lddts)
