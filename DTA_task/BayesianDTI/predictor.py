from scipy.stats import t, norm
from abc import ABC, abstractmethod
import numpy as np
import torch

device = 'cuda'


class DTIPredictor(ABC):
    """
    Abstract base class of DTI predictors.
    
    See the anoatation of self.prediction()
    
    Note that __call__ will be self.prediction()
    """
    def __call__(self, dataloader, model, **kwargs):
        return self.prediction(dataloader, model, **kwargs)
        
    
    def prediction(self, dataloader, model, **kwargs):
        """
        prediction with given dataloader and model.
        This method always returns 4 np.arrays including the mean, std of
        both predictions and ground truths although the true std label does not
        exist.
        If the model will not predict uncertainty(std); the label of std will not given,
        the return np.arrays are just a empty array.
        
        Returns:
            self.mu_pred
            self.std_pred
            self.mu_Y
            self.std_Y
        """
        self.mu_pred = []
        self.std_pred = []
        self.mu_Y = []
        self.std_Y = []
        model.eval()
        self.preconditions()
        
        with torch.no_grad():
            self.eval_loop(dataloader, model, **kwargs)
        
        self.mu_pred = np.concatenate(self.mu_pred)
        if len(self.std_pred) != 0:
            self.std_pred = np.concatenate(self.std_pred)
        self.mu_Y = np.concatenate(self.mu_Y)
        if len(self.std_Y) != 0:
            self.std_Y = np.concatenate(self.std_Y)
        
        return self.mu_pred, self.std_pred, self.mu_Y, self.std_Y

    @abstractmethod
    def eval_loop(self, dataloader, model, **kwargs):
        pass
    
    def preconditions(self):
        pass


class VanillaDTIPredictor(DTIPredictor):
    def eval_loop(self, dataloader, model, **kwargs):
        model.eval()
        predictions = []
        labels = []
        for d, p, y in dataloader:
            prediction = model(d.to(device), p.to(device))
            self.mu_pred.append(prediction.view(-1).detach().cpu().numpy())
            self.mu_Y.append(y.view(-1).detach().cpu().numpy())
    

class EviNetDTIPredictor(DTIPredictor):
    """
    Predictor class for the Evidential network.
    
    
    self.prediction(dataloader, model)
        Args:
            dataloader : Pytorch dataloader
            model: Priornet model

        return:
            mu(np.array) : Predicted affinity values.

            std(dict(np.array)) : A dictionary contains different uncertainties.
                - std['epistemic']: Epistemic uncertainty of predictions
                - std['alatoric']: Aleatoric uncertainty of predictions
                - std['total']: Total uncertainty of predictions

            mu_Y(np.array) : True affinity values.

            std_Y(np.array) : True affinity uncertainty. If not given, it is a empty list.
    """
    def eval_loop(self, dataloader, model, mcdropout=False, **kwargs):
        model.eval()
        
        alea_list = []
        epi_list = []
        total_list = []
        
        for d, p, y in dataloader:
            gamma, nu, alpha, beta = model(d.to(device), p.to(device), **kwargs)
            
            aleatoric = beta/(alpha - 1)
            epistemic = beta/(nu*(alpha - 1))
            aleatoric = aleatoric.detach().cpu().numpy().flatten()
            epistemic = epistemic.detach().cpu().numpy().flatten()
            
            total = np.sqrt(((beta*(1+nu))/(nu*alpha)).detach().cpu().numpy().flatten())
            aleatoric = np.sqrt(aleatoric)
            epistemic = np.sqrt(epistemic)
            
            self.freedom.append(alpha.view(-1).detach().cpu().numpy()*2)
            self.mu_pred.append(gamma.view(-1).detach().cpu().numpy())
            alea_list.append(aleatoric)
            epi_list.append(epistemic)
            total_list.append(total)
            self.mu_Y.append(y.view(-1).detach().cpu().numpy().flatten())
        
        self.std_pred = {
            'epistemic': np.concatenate(epi_list),
            'aleatoric': np.concatenate(alea_list),
            'total': np.concatenate(total_list)
        }
        
    def prediction(self, dataloader, model, **kwargs):
        self.mu_pred = []
        self.mu_Y = []
        self.freedom = []
        
        model.eval()
        self.preconditions()
        
        with torch.no_grad():
            self.eval_loop(dataloader, model, **kwargs)
        
        self.mu_pred = np.concatenate(self.mu_pred)
        self.freedom = np.concatenate(self.freedom)
        self.mu_Y = np.concatenate(self.mu_Y)
        
        return self.mu_pred, self.std_pred, self.mu_Y, self.freedom