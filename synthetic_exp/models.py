import torch
import numpy as np

class EvidentialNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(EvidentialNetwork, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fully2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fully3 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        self.gamma = torch.nn.Linear(hidden_dim, 1)
        self.nu = torch.nn.Linear(hidden_dim, 1)
        self.alpha = torch.nn.Linear(hidden_dim, 1)
        self.beta = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, src):
        fully1 = torch.nn.Tanh()(self.fully1(src))
        fully2 = torch.nn.Tanh()(self.fully2(fully1))
        fully3 = torch.nn.Tanh()(self.fully3(fully2))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3)) + 1 #+ 1E-9
        beta = torch.nn.Softplus()(self.beta(fully3))
        nu = torch.nn.Softplus()(self.nu(fully3))# + 1E-9
        
        gamma.retain_grad()
        nu.retain_grad()
        alpha.retain_grad()
        beta.retain_grad()
        
        return gamma, nu, alpha, beta


class Network(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(Network, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fully2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fully3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gamma = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, src):
        self.fully1_val = self.dropout(torch.nn.Tanh()(self.fully1(src)))
        self.fully2_val = self.dropout(torch.nn.Tanh()(self.fully2(self.fully1_val)))
        self.fully3_val = self.dropout(torch.nn.Tanh()(self.fully3(self.fully2_val)))
        
        gamma = self.gamma(self.fully3_val)
        return gamma
    
    def forward_s(self, src, scale=1, bias=0, s=30):
        results = []
        self.train()
        for i in range(s):
            results.append(self.forward(src)*scale + bias)
        results = torch.stack(results)
        self.eval()
        return torch.mean(results, axis=0), torch.std(results, axis=0)
    