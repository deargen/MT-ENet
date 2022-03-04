import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import torch
from mtevi.mtevi import *
from mtevi.utils import *
from models import *
import math

use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor 
                                             if torch.cuda.is_available() and x 
                                             else torch.FloatTensor)
#use_gpu()

def plot_fig(model, output='mtnet_test.png'):
    true_y = []
    pred_y = []
    std_y = []
    epi_y = []
    alea_y = []
    freedom_y = []

    for x in test_x:
        x = torch.Tensor([x])
        true_y = true_y + list(y.cpu().numpy().flatten())
        gamma, nu, alpha, beta = model(x.float())
        std_y += list(np.sqrt((beta*(1+nu)/(alpha*nu)).cpu().detach().numpy().flatten()))
        alea_y += list(np.sqrt((beta/(alpha)).cpu().detach().numpy().flatten()))
        epi_y += list(np.sqrt((beta/((alpha)*nu)).cpu().detach().numpy().flatten()))
        freedom_y += list((2*alpha).cpu().detach().numpy().flatten())
        pred_y = pred_y + list(gamma.cpu().detach().numpy().flatten())
        
    conf = confidence_interval(np.array(pred_y),
                    np.array(std_y),
                    np.array(freedom_y),
                    0.95)

    fig = plt.figure(figsize=(10,10))
    plt.scatter(X,Y, color='black')
    plt.plot(test_x, pred_y, linewidth=4)
    plt.fill_between(np.linspace(*train_boundary_A,1000), -3, 10, color='blue', alpha=.20)
    plt.fill_between(np.linspace(*train_boundary_B,1000), -3, 10, color='blue', alpha=.20)
    plt.ylim(bottom=-3, top=10)
    plt.vlines(6, -3, 10, linestyle=':', color='green', linewidth=5)

    conf = confidence_interval(np.array(pred_y),
                    np.array(std_y),
                    np.array(freedom_y), 0.90)
    plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

    conf = confidence_interval(np.array(pred_y),
                    np.array(std_y),
                    np.array(freedom_y), 0.70)
    plt.fill_between(np.linspace(-3,10,1000),conf[0], conf[1], color='red', alpha=.15)

    conf = confidence_interval(np.array(pred_y),
                    np.array(std_y),
                    np.array(freedom_y), 0.50)
    plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

    conf = confidence_interval(np.array(pred_y),
                    np.array(std_y),
                    np.array(freedom_y), 0.30)
    plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

    plt.tight_layout()
    plt.savefig(output)


np.random.seed(0)
X = np.concatenate((np.linspace(-3, 6, 1950), np.linspace(6, 10, 50)))
rand_num = [6,3]
Y = np.sin(X*4)**3 + (X**2)/10 + np.random.randn(*X.shape)*0.1 + 2*np.random.normal(scale=np.abs(7-np.abs(X))*0.05)
X = np.expand_dims(X, 1)
dim = len(X[0])

sparse_idx_A = 1970
sparse_idx_B = 1990

X_t = np.concatenate((X[400:1300], X[sparse_idx_A:sparse_idx_B]))
X_v = np.concatenate((X[:400], X[1300:sparse_idx_A], X[sparse_idx_B:]))
Y_t = np.concatenate((Y[400:1300], Y[sparse_idx_A:sparse_idx_B]))
Y_v = np.concatenate((Y[:400], Y[1300:sparse_idx_A], Y[sparse_idx_B:]))

Y_t = np.expand_dims(Y_t, 1)
Y_v = np.expand_dims(Y_v, 1)

data = []
for i in range(len(X_t)):
    data.append((X_t[i], Y_t[i]))
data_v = []
for i in range(len(X_v)):
    data_v.append((X_v[i], Y_v[i]))
    
train_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(data_v, batch_size=256, shuffle=True)

train_boundary_A = (float(X[400]), float(X[1300]))
train_boundary_B = (float(X[sparse_idx_A]), float(X[sparse_idx_B]))

test_x = np.linspace(-3,10,1000)

fig = plt.figure(figsize=(10,10))
plt.scatter(X,Y, marker='.')
plt.savefig("synthetic_result/raw_data.png")

##################################################
### MT evi net  ##################################
##################################################
##################################################
model = EvidentialNetwork(dim)
objective = EvidentialnetMarginalLikelihood()
objective_mse = torch.nn.MSELoss(reduction='none')
reg = EvidenceRegularizer(factor=0.0001)

gamma_history = []
nu_history = []
alpha_history = []
beta_history = []

mse_history = []
mse_history_v = []
nll_history = []
nll_history_v = []

total_mse = 0.
total_valid_mse = 0.
total_nll = 0.
total_valid_nll = 0.

model.train()

opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1E-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
for epoch in range(500):
    model.train()
    for x, y in train_loader:
        gamma, nu, alpha, beta = model(x.float())
        y = y.float()
        opt.zero_grad()
        
        nll = (objective(gamma,nu,alpha,beta,y)).mean()
        nll += (reg(gamma, nu, alpha, beta, y)).mean()
        total_nll += nll.item()

        mse = modified_mse(gamma, nu, alpha, beta, y).mean() 
        total_mse += mse.item()
        loss = nll + mse
        loss.backward()
        opt.step()
    
    cur_mse = total_mse/len(train_loader)
    cur_nll = total_nll/len(train_loader)
    
    model.eval()
    
    for x_v, y_v in valid_loader:
        gamma, nu, alpha, beta = model(x_v.float())
        loss_v = objective_mse(gamma, y_v).mean()
        total_valid_mse += loss_v.item()
        
        loss_v = objective(gamma, nu, alpha, beta, y_v).mean()
        total_valid_nll += loss_v.item()
        
    cur_valid_mse = total_valid_mse/len(valid_loader)
    cur_valid_nll = total_valid_nll/len(valid_loader)
    
    print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}] Train NLL [{:.5f}] Val NLL [{:.5f}]".format(
        epoch+1, cur_mse, cur_valid_mse, cur_nll, cur_valid_nll))
    mse_history_v.append(cur_valid_mse)
    nll_history_v.append(cur_valid_nll)
    total_mse = 0.
    total_valid_mse = 0.
    total_nll = 0.
    total_valid_nll = 0.
    scheduler.step()
plot_fig(model, "synthetic_result/mtnet.png")
##########################################################################
######## GP regression    ################################################
##########################################################################
##########################################################################
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
krn = 1.0*Matern(length_scale_bounds='fixed', nu=0.05)
gpr = GaussianProcessRegressor(kernel=krn, random_state=1000).fit(X_t,Y_t)
test_x_gp = np.expand_dims(np.linspace(-3,10,1000), axis=1)

pred_y, std_y = gpr.predict(test_x_gp, return_std=True)
pred_y = pred_y.flatten()

conf = confidence_interval(pred_y, std_y, None, 0.95)

fig = plt.figure(figsize=(10,10))
plt.scatter(X,Y, color='black')
plt.plot(test_x_gp, pred_y, linewidth=4)
plt.fill_between(np.linspace(*train_boundary_A,1000), -3, 10, color='blue', alpha=.20)
plt.fill_between(np.linspace(*train_boundary_B,1000), -3, 10, color='blue', alpha=.20)
plt.ylim(bottom=-3, top=10)
plt.vlines(6, -3, 10, linestyle=':', color='green', linewidth=5)


conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.90)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.70)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.50)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.30)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

plt.tight_layout()
plt.savefig("synthetic_result/gp_regression.png")
##################################################
### vanilla evi net  #############################
##################################################
##################################################
model = EvidentialNetwork(dim)
objective = EvidentialnetMarginalLikelihood()
objective_mse = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1E-3)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)

total_mse = 0.
total_valid_mse = 0.
total_nll = 0.
total_valid_nll = 0.

for epoch in range(500):
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        gamma, nu, alpha, beta = model(x.float())
        
        loss = objective(gamma, nu, alpha, beta, y).mean()
        loss += reg(gamma, nu, alpha, beta, y).mean()
        nll_history.append(loss.item())
        total_nll += loss.item()
        
        loss.backward()
        opt.step()
    
    cur_mse = total_mse/len(train_loader)
    cur_nll = total_nll/len(train_loader)
    
    model.eval()
    
    for x_v, y_v in valid_loader:
        gamma, nu, alpha, beta = model(x_v.float())
        loss_v = objective_mse(gamma, y_v)
        
        mse_history_v.append(loss_v.item())
        total_valid_mse += loss_v.item()
        
        loss_v = objective(gamma, nu, alpha, beta, y_v).mean()
        nll_history_v.append(loss_v.item())
        total_valid_nll += loss_v.item()
        
    cur_valid_mse = total_valid_mse/len(valid_loader)
    cur_valid_nll = total_valid_nll/len(valid_loader)
    
    print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}] Train NLL [{:.5f}] Val NLL [{:.5f}]".format(
        epoch+1, cur_mse, cur_valid_mse, cur_nll, cur_valid_nll))
    
    total_mse = 0.
    total_valid_mse = 0.
    total_nll = 0.
    total_valid_nll = 0.
    scheduler.step()
##################################################
true_y = []
pred_y = []
std_y = []
epi_y = []
alea_y = []
freedom_y = []

for x in test_x:
    x = torch.Tensor([x])
    true_y = true_y + list(y.cpu().numpy().flatten())
    gamma, nu, alpha, beta = model(x.float())
    std_y += list(np.sqrt((beta*(1+nu)/(alpha*nu)).cpu().detach().numpy().flatten()))
    alea_y += list(np.sqrt((beta/(alpha)).cpu().detach().numpy().flatten()))
    epi_y += list(np.sqrt((beta/((alpha)*nu)).cpu().detach().numpy().flatten()))
    freedom_y += list((2*alpha).cpu().detach().numpy().flatten())
    pred_y = pred_y + list(gamma.cpu().detach().numpy().flatten())
    
conf = confidence_interval(np.array(pred_y),
                np.array(std_y),
                np.array(freedom_y),
                0.95)

fig = plt.figure(figsize=(10,10))
plt.scatter(X,Y, color='black')
plt.plot(test_x, pred_y, linewidth=4)
plt.fill_between(np.linspace(*train_boundary_A,1000), -3, 10, color='blue', alpha=.20)
plt.fill_between(np.linspace(*train_boundary_B,1000), -3, 10, color='blue', alpha=.20)
plt.ylim(bottom=-3, top=10)
plt.vlines(6, -3, 10, linestyle=':', color='green', linewidth=5)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 np.array(freedom_y), 0.90)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 np.array(freedom_y), 0.70)
plt.fill_between(np.linspace(-3,10,1000),conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 np.array(freedom_y), 0.50)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 np.array(freedom_y), 0.30)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

plt.tight_layout()
plt.savefig("synthetic_result/evinet.png")
##########################################################################
##### MC-Dropout #########################################################
##########################################################################
##########################################################################
mse_history = []
mse_history_mc = []
nll_history_mc = []

total_mse = 0.
total_valid_mse = 0.
total_valid_nll = 0.

model = Network(dim)

objective_mse = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.005)#, weight_decay=1E-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.99)
model.train()
for epoch in range(500):
    for x, y in train_loader:
        model.train()
        opt.zero_grad()
        mu = model(x.float())

        loss = objective_mse(mu, y.float())
        mse_history.append(loss.item())
        total_mse += loss.item()
        loss.backward()
        opt.step()

        model.eval()
    cur_mse = total_mse/len(train_loader)
    model.eval()
    for x_v, y_v in valid_loader:
        mu, std = model.forward_s(x_v.float(), s=5)
        loss_v = objective_mse(mu, y_v)
        total_valid_mse += loss_v.item()
        mu = mu.cpu().detach().numpy()
        std = std.cpu().detach().numpy()
        y_v = y_v.cpu().detach().numpy() 
        loss_v = -log_likelihood(y_v, mu, std).mean()
        total_valid_nll += loss_v.item()

    cur_valid_mse = total_valid_mse/len(valid_loader)
    cur_valid_nll = total_valid_nll/len(valid_loader)

    print("Epoch {}: Train loss [{:.5f}] Val loss [{:.5f}] Val NLL [{:.5f}]".format(
        epoch+1, cur_mse, cur_valid_mse, cur_valid_nll))

    if cur_valid_nll != math.nan:
        mse_history_mc.append(cur_valid_mse)
        nll_history_mc.append(cur_valid_nll)
    total_mse = 0.
    total_valid_mse = 0.
    total_nll = 0.
    total_valid_nll = 0.
    scheduler.step()
    #############################################################################


true_y = []
pred_y = []
std_y = []
model.eval()
for x in test_x:
    x = torch.Tensor([x])
    true_y = true_y + list(y.cpu().numpy().flatten())
    mu, std = model.forward_s(x.float())
    pred_y += list(mu.cpu().detach().numpy())
    std_y += list(std.cpu().detach().numpy())
    
conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30,
                0.95)

fig = plt.figure(figsize=(10,10))
plt.scatter(X,Y, color='black')
plt.plot(test_x, pred_y, linewidth=4)
plt.fill_between(np.linspace(*train_boundary_A,1000), -3, 10, color='blue', alpha=.20)
plt.fill_between(np.linspace(*train_boundary_B,1000), -3, 10, color='blue', alpha=.20)
plt.ylim(bottom=-3, top=10)
plt.vlines(6, -3, 10, linestyle=':', color='green', linewidth=5)


conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.90)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.70)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.50)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

conf = confidence_interval(np.array(pred_y),
                 np.array(std_y),
                 30, 0.30)
plt.fill_between(np.linspace(-3,10,1000), conf[0], conf[1], color='red', alpha=.15)

plt.tight_layout()
plt.savefig("synthetic_result/mcdrop.png")
