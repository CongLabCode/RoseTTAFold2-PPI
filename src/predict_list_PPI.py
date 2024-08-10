import os, sys, time, torch, string, zipfile, argparse
import numpy as np
from data_loader import get_msa, MSAFeaturize
from RoseTTAFoldModel  import RF2trackModule
from kinematics import xyz_to_t2d
from chemical import INIT_CRDS
from AuxiliaryPredictor import DistanceNetwork


N_cycle = 3
current_FILE = os.path.abspath(__file__)
current_DIR = os.path.dirname(current_FILE)


def prep_input(msa, L1, L2, device):
    msa_in = torch.tensor(msa)
    msa_orig = msa_in[None].long().to(device) # (B, N, L)
    idx = torch.arange(L1 + L2)
    idx[L1:] += 200
    idx = idx[None].to(device)

    chain_idx = torch.zeros((L1 + L2, L1 + L2)).long()
    chain_idx[:L1, :L1] = 1
    chain_idx[L1:, L1:] = 1
    chain_idx = chain_idx[None].to(device)

    qlen = L1 + L2
    xyz_t = INIT_CRDS.reshape(1,1,27,3).repeat(1,qlen,1,1)[None].to(device) # (B, T, L, 27, 3)
    t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20, device=device).long(), num_classes=21).float() # all gaps
    conf = torch.zeros((1, qlen, 1), device=device).float()
    t1d = torch.cat((t1d, conf), -1)[None] # (B, T, L, D)
    mask_t = torch.full((1,qlen,27), False, device=device)[None] # (B, T, L, 27)

    mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
    mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
    mask_t_2d = mask_t_2d.float() * chain_idx.float()[:,None] # (ignore inter-chain region because we don't use any complex templates)
    t2d = xyz_to_t2d(xyz_t, mask_t_2d) # 2D features for templates (B, T, L, L, 37+6+1)

    network_input = {}
    network_input['msa_orig'] = msa_orig
    network_input['idx'] = idx
    network_input['t1d'] = t1d
    network_input['t2d'] = t2d
    network_input['same_chain'] = chain_idx
    return network_input


def _get_model_input(network_input, output_i, return_raw=False, use_checkpoint=False):
    '''
    Get network inputs for the current recycling step
    It will return dictionary of network inputs & labels for MLM tasks (mask_msa & msa_seed_gt)
    '''
    input_i = {}
    for key in network_input:
        if key == 'msa_orig':
            seq, msa_seed, msa_extra, mask_msa, msa_seed_gt = MSAFeaturize(network_input['msa_orig'], {'MAXLAT': 128, 'MAXSEQ':1024}, p_mask=-1.0)
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
    return input_i


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
model = RF2trackModule().to(device)

chk_fn = current_DIR + '/models/RF-PPI_epoch140.pt'
map_location={'cuda:%d'%0: device}
checkpoint = torch.load(chk_fn, map_location=map_location)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)

part = sys.argv[1]
name = sys.argv[2]
fp = open('step3' + part + '/' + name + '.log', 'r')
pair2L1 = {}
pair2L2 = {}
for line in fp:
    words = line.split()
    pair = words[0]
    pair2L1[pair] = int(words[1])
    pair2L2[pair] = int(words[2])
fp.close()


model.eval()
lp = open('step4' + part + '/' + name + '.log','w')
results = {}
with zipfile.ZipFile('step3' + part + '/' + name + '.zip', 'r') as zipf:
    for pair in zipf.namelist():
        L1 = pair2L1[pair]
        L2 = pair2L2[pair]
        with zipf.open(pair) as file:
            msa = []
            table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
            content = file.read()
            lines = content.decode('utf-8').split('\n')
            for line in lines[:10000]:
                if line:
                    if line[0] == '>':
                        continue
                    line = line.rstrip()
                    msa_i = line.translate(table)
                    msa.append(msa_i)

            msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8) # (Nseq, L)
            alphabet = np.array(list('ARNDCQEGHILKMFPSTWYVX-'), dtype='|S1').view(np.uint8)
            for i in range(alphabet.shape[0]):
                msa[msa == alphabet[i]] = i
            msa[msa > 21] = 21

            startT = time.time()
            network_input = prep_input(msa, L1, L2, device)

            with torch.no_grad():
                output_i = (None, None)
                for i_cycle in range(N_cycle):
                    if i_cycle < N_cycle-1:
                        return_raw=True
                    else:
                        return_raw=False
                    input_i = _get_model_input(network_input, output_i, return_raw=return_raw, use_checkpoint=False)
                    output_i = model(**input_i)
                    if i_cycle < N_cycle - 1:
                        continue

                    logit_dist = output_i[0][0]
                    prob_dist = torch.nn.Softmax(dim=1)(logit_dist)
                    p_bind = prob_dist[:,:20].sum(dim=1) # (B, L, L)
                    p_bind = p_bind*(1.0-network_input['same_chain'].float()) # (B, L, L)
                    p_bind = torch.nn.MaxPool2d(p_bind.shape[1:])(p_bind).view(-1)
                    p_bind = torch.clamp(p_bind, min=0.0, max=1.0)
                    _, N_seq, N_res = network_input['msa_orig'].shape
                    prob = np.sum(prob_dist[0].permute(1,2,0).detach().cpu().numpy()[:L1,L1:,:20], axis=-1).astype(np.float16)
                    results[pair] = prob
                    endT = time.time()
                    lp.write(pair + '\t' + str(round(p_bind.item(), 6)) + '\t' + str(N_seq) + '\t' + str(N_res) + '\t' + str(round(endT-startT, 6)) + '\n')
lp.close()
np.savez_compressed('step4' + part + '/' + name + '.npz', **results)
os.system('echo \'done\' > step4' + part + '/' + name + '.done')
