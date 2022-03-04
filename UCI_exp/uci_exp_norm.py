import sys
sys.path.append('..')
from mtevi.mtevi import *
from mtevi.utils import *
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import json
import math
import torch
import itertools
import argparse
import os
from bayes_opt import BayesianOptimization

to_np = lambda tensor: tensor.cpu().detach().numpy()
        
class EvidentialNetwork(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=50, prob=0.0):
        super(EvidentialNetwork, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, hidden_dim)
        self.gamma = torch.nn.Linear(hidden_dim, out_dim)
        self.nu = torch.nn.Linear(hidden_dim, out_dim)
        self.alpha = torch.nn.Linear(hidden_dim, out_dim)
        self.beta = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = torch.nn.Dropout(prob)

    def forward(self, src):
        self.fully1_val = self.dropout(torch.nn.ReLU()(self.fully1(src)))
        
        gamma = self.gamma(self.fully1_val)
        alpha = torch.nn.Softplus()(self.alpha(self.fully1_val)) + 1 #+ 1E-9
        beta = torch.nn.Softplus()(self.beta(self.fully1_val))
        nu = torch.nn.Softplus()(self.nu(self.fully1_val))# + 1E-9
        
        gamma.retain_grad()
        nu.retain_grad()
        alpha.retain_grad()
        beta.retain_grad()
        
        return gamma, nu, alpha, beta

    def freeze_mse_weights(self):
        self.fully1.param.require_grad_(False)
        self.gamma.param.require_grad_(False)


def get_loader(X, Y, expand_dims=True, **kwargs):
    X_t, X_v, Y_t, Y_v = train_test_split(X, Y, **kwargs)
    
    if Y.shape[-1] != 2 and expand_dims:
        Y_t = np.expand_dims(Y_t, 1)
        Y_v = np.expand_dims(Y_v, 1)

    data = []
    for i in range(len(X_t)):
        data.append((X_t[i], Y_t[i]))

    data_v = []
    for i in range(len(X_v)):
        data_v.append((X_v[i], Y_v[i]))

    train_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)#, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_v, batch_size=len(data_v), shuffle=True)#, pin_memory=True)
    
    return train_loader, valid_loader, X_t, Y_t

#############################################################################################
#############################################################################################
#############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--problem", help="UCI regression problem name. It could be one of ['yacht', 'boston','energy'\
    ,'kin8nm','navel','power','protein','wine','concrete']")
parser.add_argument("-c", "--cuda", action='store_true', help="cuda on or not", default=False)
args = parser.parse_args()


dir = 'uciexp_nll/'

if args.cuda:
    print("GPU training")
    use_gpu()
else:
    print("CPU training")
    
try:
    os.mkdir(dir)
except FileExistsError:
    print("The output directory {} is already exist.".format(dir))

if args.problem == None:
    parser.print_help()
    exit()
else:
    problem = args.problem

if problem == 'boston':
    X, Y = zscore(load_boston()['data'],axis=0),load_boston()['target']
else:
    X = zscore(np.genfromtxt('data/{}_X.csv'.format(problem), delimiter=','), axis=0)
    Y = np.genfromtxt('data/{}_Y.csv'.format(problem), delimiter=',')
Y_std, Y_mu = np.std(Y), np.mean(Y)
Y = zscore(Y, axis=0) 
X, Y = np.nan_to_num(X), np.nan_to_num(Y)

dim = len(X[0])
out_dim = 1
pareto_distance = dict()
pareto_distance['task'] = problem

epochs, eval_num, BO_init, BO_iter = 40, 20, 5, 25
lr_bound = (0.0005, 0.01)
fail_const = -5

reg_bound = (0, 0.01)
if problem=='protein':
    eval_num = 5
    lr_bound = (0.0001, 0.01)
    reg_bound = (0, 0.01)
if problem=='power':
    lr_bound = (0.0005, 0.05)

if problem=='year' or problem =='protein':
    hidden_dim = 100
else:
    hidden_dim = 50

#############################################################################################
#############################################################################################
#############################################################################################
mse_history_mtevireg = []
nll_history_mtevireg = []

for n_exp in range(eval_num):
    print("{}`th experiment with {}".format(n_exp+1, problem))
    train_loader, valid_loader, X_t, Y_t = get_loader(X,Y, test_size=0.1)
    train_loader_opt, valid_loader_opt, X_tt, Y_tt = get_loader(X_t, Y_t, test_size=0.2, expand_dims=False)
    ###########################################################################
    def mtevidential_eval(rate, decay_rate=0, final=False, reg_rate=0.0):
        if final:
            t_loader, v_loader = train_loader, valid_loader
        else:
            t_loader, v_loader = train_loader_opt, valid_loader_opt
        
        model = EvidentialNetwork(dim, out_dim, hidden_dim=hidden_dim)
        objective = EvidentialnetMarginalLikelihood()
        objective_mse = torch.nn.MSELoss(reduction='none')
        reg = EvidenceRegularizer(factor=reg_rate)
        opt = torch.optim.Adam(model.parameters(), lr=rate, weight_decay=decay_rate)
        
        best_mse, best_nll = 10000, 10000
        total_valid_mse, total_valid_nll = 0., 0
        for epoch in range(epochs):
            model.train()
            for x, y in t_loader:
                gamma, nu, alpha, beta = model(x.float())
                y = y.float()
                
                model.train()
                opt.zero_grad()
                nll = (objective(gamma,nu,alpha,beta,y)).mean()
                nll += (reg(gamma, nu, alpha, beta, y)).mean()
                #c = get_mse_coef_test(gamma, nu, alpha, beta, y)
                #mse = (objective_mse(gamma, y.float())*c).mean()
                mse = modified_mse(gamma, nu, alpha, beta, y)
                loss = nll + mse
                loss.backward()
                opt.step()
            ######################################################################

            model.eval()
            for x_v, y_v in v_loader:
                gamma, nu, alpha, beta = model(x_v.float())
                loss_v = objective_mse(gamma, y_v).mean() 
                total_valid_mse += loss_v.item()
                loss_v = objective(gamma, nu, alpha, beta, y_v).mean()
                total_valid_nll += loss_v.item()  
            cur_valid_mse = (total_valid_mse/len(v_loader)) * (Y_std**2)
            cur_valid_nll = (total_valid_nll/len(v_loader)) + np.log(Y_std)
            if cur_valid_mse < best_mse:
                best_mse = cur_valid_mse
            if cur_valid_nll < best_nll:
                best_nll = cur_valid_nll
            total_valid_mse, total_valid_nll = 0, 0
        if final:
            return best_mse, best_nll
        if math.isnan(-cur_valid_mse):
            return fail_const
        return -cur_valid_nll
    print("MT evidential net with reg")
        
    mtevidential_reg_eval = lambda rate, reg_rate: mtevidential_eval(rate, 0, reg_rate=reg_rate)
    pbounds = {'rate': lr_bound, 'reg_rate': reg_bound}
    optimizer = BayesianOptimization(
        f=mtevidential_reg_eval,
        pbounds=pbounds,
    )
    optimizer.maximize(init_points=BO_init, n_iter=BO_iter, acq='ei')
    mse, nll = mtevidential_eval(**optimizer.max['params'], decay_rate=0, final=True)
    print("MSE [{:.5f}], NLL [{:.5f}]".format(mse,nll))
    mse_history_mtevireg.append(mse)
    nll_history_mtevireg.append(nll)
    ############################################################################################

def get_mean_std(mse, nll):
    rmse = np.sqrt(mse)
    mean_rmse, std_rmse = np.nanmean(rmse), np.nanstd(rmse)/np.sqrt(eval_num)
    mean_nll, std_nll = np.nanmean(nll), np.nanstd(nll)/np.sqrt(eval_num)
    
    return {'rmse_mean': mean_rmse, 'rmse_std': std_rmse,
            'nll_mean': mean_nll, 'nll_std': std_nll}
    
pareto_distance['mtnet_reg'] = get_mean_std(mse_history_mtevireg, nll_history_mtevireg)
with open(dir + '{}_pareto.json'.format(problem),'w') as outfile:
    json.dump(pareto_distance, outfile)