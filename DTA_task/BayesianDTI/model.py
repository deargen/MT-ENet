"""
"""

import pickle
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from BayesianDTI.datahelper import *
from abc import ABC, abstractmethod
import torch
import torch.distributions.studentT as studentT

##################################################################################################################
#
# Abstract classes for torch models.
#
##################################################################################################################

class AbstractMoleculeEncoder(ABC, torch.nn.Module):
    """
    Abstract base class of molecule embedding models.
    """
    
    def forward(self, src):
        emb = None
        return emb

    
class AbstractProteinEncoder(ABC, torch.nn.Module):
    """
    Abstract base class of protein embedding models.
    """
    
    def forward(self, src):
        emb = None
        return emb


class AbstractInteractionModel(ABC, torch.nn.Module):
    """
    Abstract base class of drug-target interaction models.
    """
    def forward(self, protein_emb, drug_emb):
        prediction = None
        return prediction
    
    
class AbstractDTIModel(ABC, torch.nn.Module):
    def __init__(self):
        super(AbstractDTIModel, self).__init__()
        self.protein_encoder = AbstractMoleculeEncoder()
        self.smiles_encoder = AbstractProteinEncoder()
        self.interaction_predictor = AbstractInteractionModel()
        
    def forward(self, d, p):
        """
        Args:
            d(Tensor) : Preprocessed drug input batch
            p(Tensor) : Preprocessed protein input batch
            
            both d and p contains Long elements representing the token,
            such as
            ["C", "C", "O", "H"] -> Tensor([4, 4, 5, 7])
            ["P, K"] -> Tensor([12, 8])
            
        Return:
            (Tensor) [batch_size, 1]: predicted affinity value
        """
        p_emb = self.protein_encoder(p)
        d_emb = self.smiles_encoder(d)
        
        return self.interaction_predictor(p_emb, d_emb)
    
##################################################################################################
##################################################################################################
##################################################################################################


class SMILESEncoder(AbstractMoleculeEncoder):
    def __init__(self, smile_len=64+1, latent_len=128): ## +1 for 0 padding
        super(SMILESEncoder, self).__init__()
        self.encoder = torch.nn.Embedding(smile_len, latent_len)
        self.conv1 = torch.nn.Conv1d(latent_len, 32, 4)
        self.conv2 = torch.nn.Conv1d(32, 64, 6)
        self.conv3 = torch.nn.Conv1d(64, 96, 8)
        
        torch.nn.init.kaiming_normal_(self.encoder.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.encoder.weight)
        
    def forward(self, src):
        emb = self.encoder(src)
        conv1 = torch.nn.ReLU()(self.conv1(emb.transpose(1,2)))
        conv2 = torch.nn.ReLU()(self.conv2(conv1))
        conv3 = torch.nn.ReLU()(self.conv3(conv2))
        
        return torch.max(conv3, 2)[0]
        
        
class ProteinEncoder(AbstractProteinEncoder):
    def __init__(self, protein_len=25+1, latent_len=128): ## +1 for 0 padding
        super(ProteinEncoder, self).__init__()
        self.encoder = torch.nn.Embedding(protein_len, latent_len)
        self.conv1 = torch.nn.Conv1d(latent_len, 32, 4)
        self.conv2 = torch.nn.Conv1d(32, 64, 8)
        self.conv3 = torch.nn.Conv1d(64, 96, 12)
        
        torch.nn.init.kaiming_normal_(self.encoder.weight)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.encoder.weight)
    
    def forward(self, src):
        emb = self.encoder(src)
        conv1 = torch.nn.ReLU()(self.conv1(emb.transpose(1,2)))
        conv2 = torch.nn.ReLU()(self.conv2(conv1))
        conv3 = torch.nn.ReLU()(self.conv3(conv2))
        
        return torch.max(conv3, 2)[0]
    
    
class InteractionPredictor(AbstractInteractionModel):
    def __init__(self, input_dim):
        super(InteractionPredictor, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, 1024)
        self.fully2 = torch.nn.Linear(1024, 1024)
        self.fully3 = torch.nn.Linear(1024, 512)
        self.output = torch.nn.Linear(512, 1)
        self.dropout = torch.nn.Dropout(0.1)
    
    
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = torch.nn.ReLU()(self.fully1(src))
        fully1 = self.dropout(fully1)
        fully2 = torch.nn.ReLU()(self.fully2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
        return self.output(fully3)
    
    
class EvidentialLinear(torch.nn.Module):
    """
    *Note* The layer should be putted at the final of the model architecture.
    
    The output of EvidentialLineary layer is parameters of Normal-Inverse-Gamma (NIG) distribution.
    We can generate the probability distribution of the target value by using the output of this layer.
    
    The inverse-normal-gamma distribution can be formulated:
    y ~ Normal(mu, sigma**2)
    mu ~ Normal(gamma, (T/nu)**2)
    sigma ~ InverseGamma(alpha, beta)
    
    where y is a target value such as a durg-target affinity value.

    However, when we train the Evidential network and predict target values by the model,
    we do not directly use the NIG distribution. Our output probability distribution is the distribution
    by analytically marginalizing out mu and sigma[(https://arxiv.org/pdf/1910.02600); equation 6, 7].

    ************************************************************************************************
    *** Target probability distribution:
    *** p(y|gamma, nu, alpha, beta) = t-distribution(gamma, beta*(1+nu)/(nu*alpha) , 2*alpha)
    *** 
    *** We can train and infer the true value "y" by using the above probability distribution.
    ************************************************************************************************

    Args:
        gamma(Tensor): The parameter of the NIG distribution. This is the predictive value (predictive mean)
            of the output distribution.
        nu(Tensor): The parameter of the NIG distribution.
        alpha(Tensor): The parameter of the NIG distribution.
        beta(Tensor): The parameter of the NIG distribution.
    """
    def __init__(self, input_dim, output_dim=1):
        """[summary]

        Args:
            input_dim ([type]): [description]
            output_dim (int, optional): [description]. Defaults to 1.
        """
        self.gamma = torch.nn.Linear(input_dim, output_dim)
        self.nu = torch.nn.Linear(input_dim, output_dim)
        self.alpha = torch.nn.Linear(input_dim, output_dim)
        self.beta = torch.nn.Linear(input_dim, output_dim)

    def forward(self, src):
        gamma = self.gamma(src)
        alpha = torch.nn.Softplus()(self.alpha(src)) + 1
        beta = torch.nn.Softplus()(self.beta(src))
        nu = torch.nn.Softplus()(self.nu(src))

        return gamma, alpha, beta, nu


class InteractionEvidentialNetwork(AbstractInteractionModel):
    """
    Deep evidential regression - (https://arxiv.org/pdf/1910.02600)
    
    Interaction layers using Deep-evidential regression. The output neurons of this network
    are the parameter of Inverse-Normal-Gamma distribution, which is the conjugate prior of Normal.
    
    The inverse-normal-gamma distribution can be formulated:
    X ~ N(gamma, T/nu)
    T ~ InverseGamma(alpha, beta)
    
    So we can make t-distribution distribution using the parameters {gamma, nu, alpha, beta}, which are
    the output of this network.
    """
    def __init__(self, input_dim):
        super(InteractionEvidentialNetwork, self).__init__()
        self.fully1 = torch.nn.Linear(input_dim, 1024)
        self.fully2 = torch.nn.Linear(1024, 1024)
        self.fully3 = torch.nn.Linear(1024, 512)
        
        self.gamma = torch.nn.Linear(512, 1)
        self.nu = torch.nn.Linear(512, 1)#, bias=False)
        self.alpha = torch.nn.Linear(512, 1)#, bias=False)
        self.beta = torch.nn.Linear(512, 1)#, bias=False)

    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = torch.nn.ReLU()(self.fully1(src))
        fully2 = torch.nn.ReLU()(self.fully2(fully1))
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3)) + 1
        beta = torch.nn.Softplus()(self.beta(fully3))
        nu = torch.nn.Softplus()(self.nu(fully3))
        
        return gamma, nu, alpha, beta

    
class InteractionEvidentialNetworkDropout(InteractionEvidentialNetwork):
    """[summary]

    Args:
        InteractionEvidentialNetwork ([type]): [description]
    """
    def __init__(self, input_dim):
        super(InteractionEvidentialNetworkDropout, self).__init__(input_dim)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, protein_emb, drug_emb):
        src = torch.cat((protein_emb, drug_emb), 1)
        fully1 = self.dropout(torch.nn.ReLU()(self.fully1(src)))
        fully2 = self.dropout(torch.nn.ReLU()(self.fully2(fully1)))
        fully3 = torch.nn.ReLU()(self.fully3(fully2))
        
        gamma = self.gamma(fully3)
        alpha = torch.nn.Softplus()(self.alpha(fully3)) + 1
        beta = torch.nn.Softplus()(self.beta(fully3))
        nu = torch.nn.Softplus()(self.nu(fully3))
        
        return gamma, nu, alpha, beta


class DeepDTA(AbstractDTIModel):
    """
    The final DeepDTA model includes the protein encoding model;
    the smiles(drug; chemical) encoding model; the interaction model.
    """
    def __init__(self, concat_dim=96*2):
        super(DeepDTA, self).__init__()
        self.protein_encoder = ProteinEncoder()
        self.smiles_encoder = SMILESEncoder()
        self.interaction_predictor = InteractionPredictor(concat_dim)
        
    def forward(self, d, p):
        """
        Args:
            d(Tensor) : Preprocessed drug input batch
            p(Tensor) : Preprocessed protein input batch
            
            both d and p contains Long elements representing the token,
            such as
            ["C", "C", "O", "H"] -> Tensor([4, 4, 5, 7])
            ["P, K"] -> Tensor([12, 8])
            
        Return:
            (Tensor) [batch_size, 1]: predicted affinity value
        """
        p_emb = self.protein_encoder(p)
        d_emb = self.smiles_encoder(d)
        
        return self.interaction_predictor(p_emb, d_emb)
    
    def train_dropout(self):
        def turn_on_dropout(m):
            if type(m) == torch.nn.modules.dropout.Dropout:
                m.train()
        self.apply(turn_on_dropout)
    
    
class DeepDTAAleatoricBayes(AbstractDTIModel):
    """
    DeepDTA model with Aleatoric uncertainty modeling using the
    unimodel Gaussian output, which can model the noise(uncertaitny) of data itself.
    """
    def __init__(self, concat_dim=96*2):
        super(DeepDTAAleatoricBayes, self).__init__()
        self.protein_encoder = ProteinEncoder()
        self.smiles_encoder = SMILESEncoder()
        self.interaction_predictor = InteractionAleatoricUncertaintyPredictor(concat_dim)
        
    def forward(self, d, p):
        """
        Args:
            d(Tensor) : Preprocessed drug input batch
            p(Tensor) : Preprocessed protein input batch
        """
        p_emb = self.protein_encoder(p)
        d_emb = self.smiles_encoder(d)
        
        return self.interaction_predictor(p_emb, d_emb)


class EvidentialDeepDTA(DeepDTA):
    """
    DeepDTA model with Prior interaction networks.
    
    """
    def __init__(self, concat_dim=96*2, dropout=True, mtl=False):
        super(EvidentialDeepDTA, self).__init__(concat_dim=concat_dim)
        if dropout:
            self.interaction_predictor = InteractionEvidentialNetworkDropout(concat_dim)
            if mtl:
                self.interaction_predictor = InteractionEvidentialNetworkMTL(concat_dim)
        else:
            self.interaction_predictor = InteractionEvidentialNetwork(concat_dim)
    
    def forward(self, d, p):
        output_tensors = super().forward(d, p)
            
        return output_tensors
    
    @staticmethod
    def aleatoric_uncertainty(nu, alpha, beta):
        return torch.sqrt(beta/(alpha-1))
    
    @staticmethod
    def epistemic_uncertainty(alpha, beta):
        return torch.sqrt(beta/(nu*(alpha-1)))
    
    @staticmethod
    def total_uncertainty(nu,alpha,beta):
        """
        Return standard deviation of Generated student t distribution,
        
        p(y|gamma, nu, alpha, beta)  = Student-t(y; gamma, beta*(1+nu)/(nu*alpha), 2*alpha )
        
        Note that the freedom of given student-t distribution is 2alpha.
        """
        return torch.sqrt(beta*(1+nu)/(nu*alpha))

    def predictive_entropy(self, nu, alpha, beta):
        scale = (beta*(1+nu)/(nu*alpha)).sqrt()
        df = 2*alpha
        dist = studentT.StudentT(df = df, scale=scale)
        return dist.entropy()

    @staticmethod
    def freedom(alpha):
        return 2*alpha
    