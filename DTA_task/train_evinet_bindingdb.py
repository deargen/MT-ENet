import sys
sys.path.append('..')
from mtevi.mtevi import *
from mtevi.utils import *
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
parser.add_argument("--abl", action='store_true',
                    help="Use the vanilla MSE")
parser.add_argument("--evi", action='store_true',
                    help="Use the vanilla Evidential network")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")
parser.add_argument("--reg", type=float, default=1e-3,
                    help="Evidential regularization")
parser.add_argument("--model", help="Trained model")
parser.add_argument("--seed", type=int, default=0, help="The epoch to start Multi-task training. Default: 0")
parser.add_argument("--cuda", type=int, default=0, help="cuda device number")

args = parser.parse_args()
torch.cuda.set_device(args.cuda)
dir = args.output
print("Arguments: ########################")
print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
print("###################################")

try:
    os.mkdir(args.output)
except FileExistsError:
    print("The output directory {} is already exist.".format(args.output))
  
#######################################################################
### Load data
"""
Modify this part of codes to use other dataset.
"""
#######################################################################
print("Read csv data")
raw_data = pd.read_csv('data/bindingdb_ic50_nokinase.csv', header=None, na_filter=False, low_memory=False).values.T

print("Parse SMILES")
smiles = np.array(parse_molecule(list(raw_data[0]), f_format='list'))
print("Parse proteins sequences")
proteins = np.array(parse_protein(list(raw_data[1]), f_format='list', prot_len=1000))
Y = np.array(raw_data[2])
del raw_data
idx = list(range(len(Y)))
random.seed(args.seed)
random.shuffle(idx)
N = len(idx)
train_idx, valid_idx, test_idx = idx[:int(N*0.8)], idx[int(N*0.8):int(N*0.9)], idx[int(N*0.9):]

dti_dataset = DTIDataset_PDB(smiles[train_idx], proteins[train_idx], Y[train_idx])
valid_dti_dataset = DTIDataset_PDB(smiles[valid_idx], proteins[valid_idx], Y[valid_idx])
test_dti_dataset = DTIDataset_PDB(smiles[test_idx], proteins[test_idx], Y[test_idx])

dataloader = DataLoader(dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset)#, pin_memory=True)
valid_dataloader = DataLoader(valid_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset)#, pin_memory=True)
test_dataloader = DataLoader(test_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset)#, pin_memory=True)
##########################################################################
### Define models
device = 'cuda:{}'.format(args.cuda)
if args.model == None:
    dti_model = EvidentialDeepDTA(dropout=True).to(device)
else:
    dti_model = torch.load(args.model).to(device)

objective_fn = EvidentialnetMarginalLikelihood()
objective_mse = torch.nn.MSELoss()

regularizer = EvidenceRegularizer(factor=args.reg).to(device)
opt = torch.optim.Adam(dti_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda epoch: 0.99 ** epoch,
                                last_epoch=-1,
                                verbose=False)

total_mse = 0.
total_valid_loss = 0.
total_nll = 0.
total_valid_nll = 0.
i = 1
loss_history = []
nll_history = []
valid_loss_history = []
valid_nll_history = []
##########################################################################
### Training
i = 0
best_nll = 10000
for epoch in range(args.epochs):
    dti_model.train()
    for d, p, y in dataloader:
        opt.zero_grad()
        y = y.unsqueeze(1).to(device)
        gamma, nu, alpha, beta = dti_model(d.to(device), p.to(device))
        
        loss = objective_fn(gamma, nu, alpha, beta, y)#.mean()
        total_nll += loss.item()
        loss += regularizer(gamma, nu, alpha, beta, y)#.mean()
        if not args.evi:
            if args.abl:
                mse = objective_mse(gamma, y)
            else:
                mse = modified_mse(gamma, nu, alpha, beta, y)
            total_mse += mse.item()
            loss += mse
        loss.backward()
        opt.step()
        i += 1
        if i % 100 == 0:
            if not args.evi:
                print("Iteration {}: Train NLL [{:.5f}] Train MSE [{:.5f}]".format(i, total_nll / i, total_mse / i))
            else:
                print("Iteration {}: Train NLL [{:.5f}]".format(i, total_nll / i))
    dti_model.eval()
    for d_v, p_v, y_v in valid_dataloader:
        gamma, nu, alpha, beta = dti_model(d_v.to(device), p_v.to(device))
        nll_v = objective_fn(gamma,
                           nu,
                           alpha,
                           beta,
                           y_v.unsqueeze(1).to(device)).mean()
        valid_nll_history.append(nll_v.item())
        total_valid_nll += nll_v.item()
        
        nll_v = objective_mse(gamma, y_v.unsqueeze(1).to(device))
        valid_loss_history.append(nll_v.item())
        total_valid_loss += nll_v.item()
        
    train_nll = total_nll/len(dataloader) 
    valid_nll = total_valid_nll/len(valid_dataloader)
    valid_loss = total_valid_loss/len(valid_dataloader)
    if math.isnan(valid_nll):
        break
    if best_nll >= valid_nll:
        torch.save(dti_model, dir + '/dti_model_best.model')
        best_nll = valid_nll
    print("Epoch {}: Train NLL [{:.5f}] Val MSE [{:.5f}] Val NLL [{:.5f}]".format(
        epoch+1, train_nll, valid_loss, valid_nll))
    total_valid_loss = 0.
    total_valid_nll = 0.
    total_nll = 0.
    total_mse = 0.
    i = 0
torch.save(dti_model, dir + '/dti_model.model')
##########################################################################
fig = plt.figure(figsize=(15,5))
plt.plot(valid_loss_history, label="MSE")
plt.plot(valid_nll_history, label="NLL")
plt.title("Validate loss")
plt.xlabel("Validate steps")
plt.legend(facecolor='white', edgecolor='black')

plt.tight_layout()

plt.savefig(dir + "/MultitaskLoss.png")
##########################################################################
### Evaluation
import torch.distributions.studentT as studentT
predictor = EviNetDTIPredictor()

eval_model = torch.load(dir + '/dti_model_best.model').to(device)
   
mu_t, std_t, mu_Y_t, freedom_t = predictor(test_dataloader, eval_model)

total_t = std_t['total']
epistemic_t = std_t['epistemic']
aleatoric_t = std_t['aleatoric']

predictive_entropy = studentT.StudentT(torch.from_numpy(freedom_t), scale=torch.from_numpy(total_t)).entropy() 
##########################################################################
from BayesianDTI.utils import plot_predictions
plot_predictions(mu_Y_t, mu_t, total_t, title="Mean prediction test with total uncertainty",
                 sample_num=freedom_t, savefig=dir + "/total_uncertainty.png")
plot_predictions(mu_Y_t, mu_t, aleatoric_t, title="Mean prediction test with aleatoric uncertainty",
                 sample_num=None, savefig=dir + "/aleatoric_uncertainty.png")
plot_predictions(mu_Y_t, mu_t, epistemic_t, title="Mean prediction test with epistemic uncertainty",
                 sample_num=None, savefig=dir + "/epistemic_uncertainty.png")
plot_predictions(mu_Y_t, mu_t, predictive_entropy, title="Mean prediction test with total uncertainty",
                 sample_num=freedom_t, 
                 savefig=dir + "/total_uncertainty_colored.png", rep_conf='color')
##########################################################################
from BayesianDTI.utils import evaluate_model
import json

eval_results = evaluate_model(mu_Y_t, mu_t, total_t, sample_num=freedom_t)
eval_json = json.dumps(eval_results, indent=4)
print(eval_json)
with open(dir + '/eval_result_prior.json','w') as outfile:
    json.dump(eval_results, outfile)
