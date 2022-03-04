import numpy as np
import torch

def modified_mse(gamma, nu, alpha, beta, target, reduction='mean'):
    """
    Lipschitz MSE loss of the "Improving evidential deep learning via multi-task learning."

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        target ([FloatTensor]): true labels.
        reduction (str, optional): . Defaults to 'mean'.

    Returns:
        [FloatTensor]: The loss value. 
    """
    mse = (gamma-target)**2
    c = get_mse_coef(gamma, nu, alpha, beta, target).detach()
    modified_mse = mse*c
    if reduction == 'mean': 
        return modified_mse.mean()
    elif reduction == 'sum':
        return modified_mse.sum()
    else:
        return modified_mse


def get_mse_coef(gamma, nu, alpha, beta, y):
    """
    Return the coefficient of the MSE loss for each prediction.
    By assigning the coefficient to each MSE value, it clips the gradient of the MSE
    based on the threshold values U_nu, U_alpha, which are calculated by check_mse_efficiency_* functions.

    Args:
        gamma ([FloatTensor]): the output of the ENet.
        nu ([FloatTensor]): the output of the ENet.
        alpha ([FloatTensor]): the output of the ENet.
        beta ([FloatTensor]): the output of the ENet.
        y ([FloatTensor]): true labels.

    Returns:
        [FloatTensor]: [0.0-1.0], the coefficient of the MSE for each prediction.
    """
    alpha_eff = check_mse_efficiency_alpha(gamma, nu, alpha, beta, y)
    nu_eff = check_mse_efficiency_nu(gamma, nu, alpha, beta, y)
    delta = (gamma - y).abs()
    min_bound = torch.min(nu_eff, alpha_eff).min()
    c = (min_bound.sqrt()/delta).detach()
    return torch.clip(c, min=False, max=1.)


def check_mse_efficiency_alpha(gamma, nu, alpha, beta, y, reduction='mean'):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for alpha, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, alpha).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial alpha(numpy.array) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    
    """
    delta = (y-gamma)**2
    right = (torch.exp((torch.digamma(alpha+0.5)-torch.digamma(alpha))) - 1)*2*beta*(1+nu) / nu

    return (right).detach()


def check_mse_efficiency_nu(gamma, nu, alpha, beta, y):
    """
    Check the MSE loss (gamma - y)^2 can make negative gradients for nu, which is
    a pseudo observation of the normal-inverse-gamma. We can use this to check the MSE
    loss can success(increase the pseudo observation, nu).
    
    Args:
        gamma, nu, alpha, beta(torch.Tensor) output values of the evidential network
        y(torch.Tensor) the ground truth
    
    Return:
        partial f / partial nu(torch.Tensor) 
        where f => the NLL loss (BayesianDTI.loss.MarginalLikelihood)
    """
    gamma, nu, alpha, beta = gamma.detach(), nu.detach(), alpha.detach(), beta.detach()
    nu_1 = (nu+1)/nu
    return (beta*nu_1/alpha)


class EvidentialnetMarginalLikelihood(torch.nn.modules.loss._Loss):
    """
    Marginal likelihood error of prior network.
    The target value is not a distribution (mu, std), but a just value.
    
    This is a negative log marginal likelihood, with integral mu and sigma.
    
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9):
        super(EvidentialnetMarginalLikelihood, self).__init__(size_average, reduce, reduction)
    
    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gamma, nu, alpha, beta -> outputs of prior network
            
            target -> target value
            
        Return:
            (Tensor) Negative log marginal likelihood of EvidentialNet
                p(y|m) = Student-t(y; gamma, (beta(1+nu))/(nu*alpha) , 2*alpha)

                then, the negative log likelihood is (CAUTION QUITE COMPLEX!)

                NLL = -log(p(y|m)) =
                    log(3.14/nu)*0.5 - alpha*log(2*beta*(1 + nu)) + (alpha + 0.5)*log( nu(target - gamma)^2 + 2*beta(1 + nu) )
                    + log(GammaFunc(alpha)/GammaFunc(alpha + 0.5))
        """
        pi = torch.tensor(np.pi)
        
        x1 = torch.log(pi/nu)*0.5
        x2 = -alpha*torch.log(2.*beta*(1.+ nu))
        x3 = (alpha + 0.5)*torch.log( nu*(target - gamma)**2 + 2.*beta*(1. + nu) )
        x4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        if self.reduction == 'mean': 
            return (x1 + x2 + x3 + x4).mean()
        elif self.reduction == 'sum':
            return (x1 + x2 + x3 + x4).sum()
        else:
            return (x1 + x2 + x3 + x4)

    
class EvidenceRegularizer(torch.nn.modules.loss._Loss):
    """
    Regularization for the regression prior network.
    If the self.factor increases, the model output the wider(high confidence interval) predictions.
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', eps=1e-9, factor=0.1):
        super(EvidenceRegularizer, self).__init__(size_average, reduce, reduction)
        self.factor = factor
    
    def forward(self, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gamma, nu, alpha, beta -> outputs of prior network
            
            target -> target value
            
            factor -> regularization strength
        
        Return:
            (Tensor) prior network regularization
            Loss = |y - gamma|*(2*nu + alpha) * factor
            
        """
        loss_value =  torch.abs(target - gamma)*(2*nu + alpha) * self.factor
        if self.reduction == 'mean': 
            return (loss_value).mean()
        elif self.reduction == 'sum':
            return (loss_value).sum()
        else:
            return (loss_value)
    

class GaussianNLL(torch.nn.modules.loss._Loss):
    """
    Negative Gaussian likelihood loss.
    """
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(GaussianNLL, self).__init__(size_average, reduce, reduction)
    
    def forward(self, input_mu: torch.Tensor, input_std: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        x1 = 0.5*torch.log(2*np.pi*input_std*input_std)
        x2 = 0.5/(input_std**2)*((target - input_mu)**2)
        
        if self.reduction == 'mean':
            return torch.mean(x1 + x2)
        elif self.reduction == 'sum':
            return torch.sum(x1 + x2)