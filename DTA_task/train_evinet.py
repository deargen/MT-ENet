import sys
sys.path.append('..')
from mtevi.mtevi import *
from mtevi.utils import *
import numpy as np
import torch
import argparse
import os
import math
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
parser.add_argument("-e", "--epochs", type=int, default=200,
                    help="Number of epochs.")
parser.add_argument("-o", "--output",
                    help="The output directory.")
parser.add_argument("--type", default='None',
                    help="Davis or Kiba; dataset select.")
parser.add_argument("--model",
                    help="The trained baseline model. If given, keep train the model.")
parser.add_argument("--abl", action='store_true',
                    help="Use the vanilla MSE")
parser.add_argument("--evi", action='store_true',
                    help="Use the vanilla evidential network")
parser.add_argument("--reg", type=float, default=0.0001,
                    help="Coefficient of evidential regularization")
parser.add_argument("--l2", type=float, default=0.0001,
                    help="Coefficient of L2 regularization")
parser.add_argument("--cuda", type=int, default=0, help="cuda device number")

args = parser.parse_args()
torch.cuda.set_device(args.cuda)
args.type = args.type.lower()
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

dataloader = DataLoader(dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset)#, pin_memory=True)
valid_dataloader = DataLoader(valid_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset)#, pin_memory=True)
test_dataloader = DataLoader(test_dti_dataset, batch_size=256, shuffle=True, collate_fn=collate_dataset)#, pin_memory=True)
##########################################################################
### Define models
device = 'cuda:{}'.format(args.cuda)

dti_model = EvidentialDeepDTA(dropout=True).to(device)

objective_fn = EvidentialnetMarginalLikelihood().to(device)
objective_mse = torch.nn.MSELoss()

regularizer = EvidenceRegularizer(factor=args.reg).to(device)
opt = torch.optim.Adam(dti_model.parameters(), lr=0.001, weight_decay=args.l2)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda epoch: 0.99 ** epoch,
                                last_epoch=-1,
                                verbose=False)

total_valid_loss = 0.
total_valid_nll = 0.
total_nll = 0.
train_nll_history = []
valid_loss_history = []
valid_nll_history = []
##########################################################################
### Training

a_history = []
best_nll = 10000
for epoch in range(args.epochs):
    dti_model.train()
    for d, p, y in dataloader:
        y = y.unsqueeze(1).to(device)
        gamma, nu, alpha, beta = dti_model(d.to(device), p.to(device))
        
        opt.zero_grad()
        ###############################################################
        #### NLL training
        loss = objective_fn(gamma, nu, alpha, beta, y)
        total_nll += loss.item()
        loss += regularizer(gamma, nu, alpha, beta, y)
        if not args.evi:
            if args.abl:
                mse = objective_mse(gamma, y)
            else:
                mse = modified_mse(gamma, nu, alpha, beta, y)
            loss += mse.mean()
        loss.backward()
        ###############################################################
        opt.step()
    scheduler.step()
    
    dti_model.eval()
    for d_v, p_v, y_v in valid_dataloader:
        y_v = y_v.unsqueeze(1).to(device)
        gamma, nu, alpha, beta = dti_model(d_v.to(device), p_v.to(device))
        nll_v = objective_fn(gamma,
                           nu,
                           alpha,
                           beta,
                           y_v)
        valid_nll_history.append(nll_v.item())
        total_valid_nll += nll_v.item()
        
        nll_v = objective_mse(gamma, y_v).mean()
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
    total_nll = 0.
    total_valid_loss = 0.
    total_valid_nll = 0.
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
plot_predictions(mu_Y_t, mu_t, predictive_entropy, title="Mean prediction test with predictive entropy",
                 sample_num=freedom_t, savefig=dir + "/total_uncertainty_colored.png", rep_conf='color')
##########################################################################
from BayesianDTI.utils import evaluate_model
import json

eval_results = evaluate_model(mu_Y_t, mu_t, total_t, sample_num=freedom_t)
eval_json = json.dumps(eval_results, indent=4)
print(eval_json)
with open(dir + '/eval_result_prior.json','w') as outfile:
    json.dump(eval_results, outfile)
