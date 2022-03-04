import numpy as np
import torch
import argparse
import os
import sys
import pandas as pd

from BayesianDTI.utils import *
from torch.utils.data import Dataset, DataLoader
from BayesianDTI.datahelper import *
from BayesianDTI.model import *
from BayesianDTI.loss import *
from BayesianDTI.predictor import *
from scipy.stats import t
from BayesianDTI.utils import confidence_interval
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="Epochs. Default: 100")
parser.add_argument("-o", "--output",
                    help="The output directory.")
parser.add_argument("--seed", type=int, default=0, help="The epoch to start Multi-task training. Default: 0")
parser.add_argument("--model", help="Trained model")
parser.add_argument("--cuda", type=int, default=0, help="cuda device number")

args = parser.parse_args()
torch.cuda.set_device(args.cuda)
dir = args.output

try:
    os.mkdir(args.output)
except FileExistsError:
    print("The output directory {} is already exist.".format(args.output))
  
#*######################################################################
#*## Load data
#*######################################################################
prot_dict = fasta2dict('data/uniprot.fasta')
raw_data = pd.read_csv('data/bindingdb_ic50_nokinase.csv', header=None, na_filter=False, low_memory=False).values.T

smiles = np.array(parse_molecule(list(raw_data[0]), f_format='list'))
proteins = np.array(parse_protein(list(raw_data[1]), f_format='list'))
Y = np.array(raw_data[2])

idx = list(range(len(Y)))
random.seed(args.seed)
random.shuffle(idx)
N = len(idx)
train_idx, valid_idx, test_idx = idx[:int(N*0.8)], idx[int(N*0.8):int(N*0.9)], idx[int(N*0.9):]

dti_dataset = DTIDataset_PDB(smiles[train_idx], proteins[train_idx], Y[train_idx])
valid_dti_dataset = DTIDataset_PDB(smiles[valid_idx], proteins[valid_idx], Y[valid_idx])
test_dti_dataset = DTIDataset_PDB(smiles[test_idx], proteins[test_idx], Y[test_idx])

dataloader = DataLoader(dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset, pin_memory=True)
valid_dataloader = DataLoader(valid_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset, pin_memory=True)
test_dataloader = DataLoader(test_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset, pin_memory=True)
##########################################################################
### Define models
device = 'cuda:{}'.format(args.cuda)
if args.model == None:
    dti_model = DeepDTA().to(device)
else:
    dti_model = torch.load(args.model).to(device)
objective_mse = torch.nn.MSELoss(reduction='none')
opt = torch.optim.Adam(dti_model.parameters(), lr=0.001)
##########################################################################
### Training
total_loss = 0.
total_valid_loss = 0.
valid_loss_history = []

a_history = []
best_mse = 10000
i = 0
for epoch in range(args.epochs):
    dti_model.train()
    for d, p, y in dataloader:
        y = y.unsqueeze(1).to(device)
        opt.zero_grad()
        pred = dti_model(d.to(device), p.to(device))
        loss = objective_mse(pred, y).mean()
        total_loss += loss.item()
        loss.backward()
        opt.step()
        i += 1
        if i % 100 == 0:
            print("Iteration {}: Train MSE [{:.5f}]".format(i, total_loss / i))
    
    dti_model.eval()
    for d_v, p_v, y_v in valid_dataloader:
        pred_v = dti_model(d_v.to(device), p_v.to(device))
        
        loss_v = objective_mse(pred_v, y_v.unsqueeze(1).to(device)).mean()
        valid_loss_history.append(loss_v.item())
        total_valid_loss += loss_v.item()
        
    valid_loss = total_valid_loss/len(valid_dataloader)
    if best_mse >= valid_loss:
        torch.save(dti_model, dir + '/dti_model_best.model')
        best_mse = valid_loss
    print("Epoch {}: Val MSE [{:.5f}]".format(epoch+1, valid_loss))
    
    total_valid_loss = 0.
    total_loss = 0.
    i = 0.
torch.save(dti_model, dir + '/dti_model.model')
##########################################################################
### Evaluation
from BayesianDTI.predictor import MCDropoutDTIPredictor
eval_model = torch.load(dir + '/dti_model_best.model')
predictor = MCDropoutDTIPredictor() 
mu_t, std_t, Y_t, _ = predictor(test_dataloader, eval_model, sample_num=5)
tau = 4
std_t = np.sqrt((std_t + 1/tau)**2)
##########################################################################
from BayesianDTI.utils import plot_predictions
plot_predictions(Y_t, mu_t, std_t, title="Mean prediction test with total uncertainty",
                 sample_num=None, savefig=dir + "/total_uncertainty.png")
plot_predictions(Y_t, mu_t, std_t, title="Mean prediction test with total uncertainty",
                 sample_num=None, 
                 savefig=dir + "/total_uncertainty_colored.png", rep_conf='color')
##########################################################################
from BayesianDTI.utils import evaluate_model
import json

eval_results = evaluate_model(Y_t, mu_t, std_t, sample_num=None, mcdrop=True)
print(eval_results)
eval_json = json.dumps(eval_results, indent=4)
print(eval_json)
with open(dir + '/eval_result.json','w') as outfile:
    json.dump(eval_results, outfile)