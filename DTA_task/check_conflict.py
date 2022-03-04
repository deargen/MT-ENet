import numpy as np
import torch
import argparse
import os
import sys
sys.path.append('..')
from mtevi.mtevi import *
from mtevi.utils import *
import pickle
import pdb
from BayesianDTI.utils import *
from torch.utils.data import Dataset, DataLoader
from BayesianDTI.datahelper import *
from BayesianDTI.model import *
from BayesianDTI.predictor import *
from scipy.stats import t
from BayesianDTI.utils import confidence_interval
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fold_num", type=int,
                    help="Fold number. It must be one of the {0,1,2,3,4}.")
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="Epochs. Default: 100")
parser.add_argument("-o", "--output", default='cos_sims.pkl',
                    help="The output pickle.")
parser.add_argument("--type", default='None',
                    help="Davis or Kiba; dataset select.")
parser.add_argument("--abl", action='store_true',
                    help="Use the vanilla MSE")
parser.add_argument("--cuda", type=int, default=0, help="cuda device number")


args = parser.parse_args()
torch.cuda.set_device(args.cuda)
args.type = args.type.lower()
#######################################################################
### Load data
FOLD_NUM = int(args.fold_num) # {0,1,2,3,4}

class DataSetting:
    def __init__(self):
        self.dataset_path = 'data/{}/'.format(args.type)
        self.problem_type = '1'
        self.is_log = False if args.type == 'kiba' else True

data_setting = DataSetting()
        
dataset = DataSet(data_setting.dataset_path,
                  1000 if args.type == 'kiba' else 1200,
                  100 if args.type == 'kiba' else 85) ## KIBA (1000,100) DAVIS (1200, 85)
smiles, proteins, Y = dataset.parse_data(data_setting)
test_fold, train_folds = dataset.read_sets(data_setting)

label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
test_drug_indices = label_row_inds[test_fold]
test_protein_indices = label_col_inds[test_fold]

train_fold_sum = []
for i in range(5):
    if i != FOLD_NUM:
        train_fold_sum += train_folds[i]

train_drug_indices = label_row_inds[train_fold_sum]
train_protein_indices = label_col_inds[train_fold_sum]

valid_drug_indices = label_row_inds[train_folds[FOLD_NUM]]
valid_protein_indices = label_col_inds[train_folds[FOLD_NUM]]

dti_dataset = DTIDataset(smiles, proteins, Y, train_drug_indices, train_protein_indices)
valid_dti_dataset = DTIDataset(smiles, proteins, Y, valid_drug_indices, valid_protein_indices)
test_dti_dataset = DTIDataset(smiles, proteins, Y, test_drug_indices, test_protein_indices)

dataloader = DataLoader(dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset, pin_memory=True)
valid_dataloader = DataLoader(valid_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset, pin_memory=True)
test_dataloader = DataLoader(test_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset, pin_memory=True)

##########################################################################
### Define models
device = 'cuda:{}'.format(args.cuda)

dti_model = EvidentialDeepDTA(dropout=False).to(device)

objective_fn = EvidentialnetMarginalLikelihood().to(device)
objective_mse = torch.nn.MSELoss(reduction='none')

regularizer = EvidenceRegularizer(factor=0.001).to(device)
opt = torch.optim.Adam(dti_model.parameters(), lr=0.001)

total_valid_loss = 0.
total_valid_nll = 0.
valid_loss_history = []
valid_nll_history = []
##########################################################################
### Training
sim = torch.nn.CosineSimilarity(dim=0)
cos_sims = []
steps = 0
best_nll = 10000
for epoch in range(args.epochs):
    dti_model.train()
    for d, p, y in dataloader:
        opt.zero_grad()
        y = y.unsqueeze(1).to(device)
        gamma, nu, alpha, beta = dti_model(d.to(device), p.to(device))
        if args.abl: 
            (objective_mse(gamma,y)*0.1).mean().backward(retain_graph=True)
        else:
            modified_mse(gamma, nu, alpha, beta, y).mean().backward(retain_graph=True)
        
        grad_mse = get_gradient_vector(dti_model)
        opt.zero_grad()
        
        objective_fn(gamma,nu,alpha, beta, y).mean().backward(retain_graph=True)
        grad_nll = get_gradient_vector(dti_model)
        opt.zero_grad()
        
        cos_sim = sim(grad_mse, grad_nll).item()
        cos_sims.append(cos_sim)
        print("[{}/10000] Cos sim :[{:.5f}]".format(steps,cos_sim))
        ###############################################################
        #### NLL training
        loss = objective_fn(gamma, nu, alpha, beta, y).mean()
        if args.abl:
            mse = objective_mse(gamma, y)
        else:
            mse = modified_mse(gamma, nu, alpha, beta, y)
        loss += (mse).mean()
            
        loss.backward()
        ###############################################################
        opt.step()
        steps += 1
        if steps >= 10000:
            break
    if steps >= 10000:
        break

pickle.dump(cos_sims, open(args.output,'wb'))
##########################################################################
