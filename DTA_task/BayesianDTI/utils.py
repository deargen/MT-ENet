"""
This source includs two types of classes or functions.

1. Evaluation methods:
    This source includs some evaluation metrics, such as r-square, CI, mse ...
    or even confidence intervals and ECE, which are uncertainty measures.
    
2. Some plotting functions.

"""
from lifelines.utils import concordance_index
from scipy import stats
from scipy.stats import t, norm, gamma
from abc import ABC, abstractmethod
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

device = 'cuda'

to_np = lambda tensor: tensor.cpu().detach().numpy()

def informative_prior(chair_model, member_model, tau=1.):
    chair_prior_loss = 0
    chair_parameters, member_parameters = chair_model.parameters(), member_model.parameters()
    for w_c, w_m in zip(chair_parameters, member_parameters):
        chair_prior_loss += ((w_c - w_m)**2/tau).mean()
    return chair_prior_loss

def clean_up_grad(tensors):
    for t in tensors:
        t.grad.data.zero_()

def gradient_scaling(model, alpha):
    """
    Scale alpha times the gradients of the models.
    
    for ever paramter w,
        dw/dl = dw/dl * alpha
    
    Args:
        model(torch.nn.Module)
        alpha(torch.nn.FloatTensor, Float)
    
    Return:
        None
    """
    for w in model.parameters():
        w.grad *= alpha
    

def eval_uncertainty(mu, std, Y, confidences=None, sample_num=10, verbose=False, overconferr=False, **kwargs):
    """
    "Accurate Uncertainties for Deep Learning Using Clibrated Regression"
    ICML 2020; Kuleshov et al. suggested the ECE-like uncertainty evaluation method
    for uncertainty estimation of regression-tasks.
    Note that the calibration metric(error) is in 3.5, Calibration section of the paper.
    
    p_j = <0.1, 0.2, ..., 0.9>
    Acc(p_j) = The accuracy of true values being inside between confidence interval "p_j"
    
    Args:
        mu(np.array): Numpy array of the predictive mean
        std(np.array): Numpy array of the predictive standard derivation
        Y(np.arry): Numpy array of ground truths
        confidences
        sample_num(Int or np.array): Number of samples to calculate t-distributions. If None, it uses Normal.
    
    Return: 
        Tuple(Metric = sum[ (p_j - ACC(p_j))^2 ], List of confidence errors for each confidence level)
    """
    if confidences == None:
        confidences = np.linspace(0, 1, num=100)
        #confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    calibration_errors = []
    interval_acc_list = []
    for confidence in confidences:
        low_interval, up_interval = confidence_interval(mu, std, sample_num, confidence=confidence)
        hit = 0
        for i in range(len(Y)):
            if low_interval[i] <= Y[i] and Y[i] <= up_interval[i]:
                hit += 1
        
        interval_acc = hit/len(Y)
        interval_acc_list.append(interval_acc)
        
        if verbose:
            print("Interval acc: {}, confidence level: {}".format(interval_acc, confidence))
        if overconferr:
            calibration_errors.append(confidence*np.maximum((confidence - interval_acc), 0))
        else:
            calibration_errors.append((confidence - interval_acc)**2)
            
    return sum(calibration_errors), calibration_errors, interval_acc_list
    

def confidence_interval(mu, std, sample_num, confidence=0.9):
    """
    Calculate confidence interval from mean and std for each predictions
    under the empricial t-distribution.
    
    If the sample_num is given as the 0, it will compute Gaussian confidence interval,
    not t-distribution
    
    If we use evidential network with following outputs
    : gamma, nu, alpha, beta = model(X)

    then we the std and mu will be
    : mu, std, freedom = gamma, (beta*(nu + 1))/(alpha*nu), 2*alpha
    : low_interval, up_interval = confidence_interval(mu, std, freedom)

    Args:
        mu(np.array): Numpy array of the predictive mean
        std(np.array): Numpy array of the predictive standard derivation
        sample_num(np.array or Int): Degree of freedom. It can be a [np.array] having same dimension
            with the mu and std.
        confidence(Float): confidence level, [0,1] (eg: 0.99, 0.95, 0.1)
    Return:
        low_interval(np.array), up_interval(np.array): confidence intervals
    """
    n = sample_num
    if type(sample_num) == np.ndarray:
        h = std * t.ppf((1 + confidence) / 2, n - 1)
    else:
        if sample_num != None:
            h = std * t.ppf((1 + confidence) / 2, n - 1)
        else:
            h = std * norm.ppf((1 + confidence) / 2)
    low_interval = mu - h
    up_interval = mu + h
    return low_interval, up_interval


def plot_predictions(y, preds, std=[], title="Predictions", savefig=None,
                     interval_level=0.9, v_max=None, v_min=None, sample_num=30,
                     rep_conf="bar", post_setting=None, **kwargs):
    """
    Show a scatter plot for prediction accuracy and if given, confidence levels.
    
    Args:
        y(np.array) ground truth predictions.
        
        preds(np.array) predicted values.
        
        std(np.array) predicted std, if given, draw confidence interval.
        
        title(str) The title of the plot.
        
        savefig(str) The name of output figure file. If not given, just show the figure.
        
        interval_level(float) The level of confidence interval, e. g) 0.9 will show 90%
            confidence interval
        
        v_max, v_min(float, float) minimum, maximum value of x, y axises.
        
        sample_num(float or np.array) Sampling number to calculate confidence level using
            t-distribution. It is a freedom parameter of t-distribution. You can assign
            different freedoms(sample_num) for each sample by using np.array as the sample_num.
            
        rep_conf('bar' or 'color') If 'bar' choosen, the confidence interval will be represented
            as bars. If 'color' choosen, it will be colors for dots.
        
        post_setting(func) This function 
        
        **kwargs -> You can pass some arguments for matplotlib plot functions.
        
    Return:
        None
    """
    if v_min == None:
        v_min = min(min(y), min(preds))*0.98
    if v_max == None:
        v_max = max(max(y), max(preds))*1.02
    
    fig = plt.figure(figsize=(10,10))
    if len(std) == 0:
        plt.scatter(y, preds, **kwargs)
    else:
        if rep_conf == 'bar':
            plt.errorbar(y, preds,
                     yerr=(confidence_interval(preds, std, sample_num, interval_level)-preds)[1],
                     fmt='o',
                     ecolor='black',
                     elinewidth=1,
                     **kwargs)
        elif rep_conf == 'color':
            fig = plt.figure(figsize=(12,10))
            plt.scatter(y, preds, c=std, **kwargs)
            plt.colorbar()
        else:
            print("rep_conf should be either 'bar' or 'color'. Not {}".format(rep_conf))
            return None
    
    plt.axline((v_min, v_min), (v_max,v_max), color='black', linestyle='--')
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.ylim(bottom=v_min, top=v_max)
    plt.title(title, fontsize=20)
    if post_setting != None:
        post_setting()
    
    if savefig != None:
        plt.savefig(savefig)
    else:
        plt.show()
        

def aupr(Y, P, data_type='kiba', **kwargs):
    from sklearn import metrics
    if data_type=='kiba':
        threshold = 12.1
    elif data_type=='davis':
        threshold = 7
    Y = np.copy(Y)
    Y[Y < threshold] = 0
    Y[Y >= threshold] = 1
    
    return metrics.average_precision_score(Y, P)
        
"""
Metrics from DeepDTA sources
"""
def get_cindex(Y, P):
    """
    ******NOTE; CAUTION******
    This code is from the original repository of "DeepDTA; Bioinformatics"
    Now the get_cindex is invalid (order dependent of given pairs).
    We will use lifelines.utils for 
    """
    summ = 0
    pair = 0
    
    for i in range(0, len(Y)):
        for j in range(0, len(Y)):
            if i != j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
            
    if pair != 0:
        return summ/pair
    else:
        return 0


def r_squared_error(y_obs,y_pred):
    """
    Calculate r^2, r-square metric, which is commonly used for QSAR evaluation.
    
    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

##################################################################

def log_likelihood(Y, preds, scale, sample_num=30, **kwargs):
    """
    Calculate negative log likelihood
    """
    #print("Log-likelihood, sample num: " + str(sample_num))
    if type(sample_num) == np.ndarray: 
        nll_t = lambda y, mu, std, freedom: np.log(t.pdf(y, freedom, mu, std))
    else:
        if sample_num == None:
            nll_t = lambda y, mu, std: np.log(norm.ppf(y, mu, std))
        else:
            nll_t = lambda y, mu, std: np.log(t.pdf(y, sample_num, mu, std))
        
    nll_values = []
    for i in range(len(Y)):
        if type(sample_num) == np.ndarray: 
            nll_values.append(nll_t(Y[i], preds[i], scale[i], sample_num[i]))
        else:
            nll_values.append(nll_t(Y[i], preds[i], scale[i]))
    return np.mean(nll_values)


def mcdrop_nll(y, preds, std, tau=1.0):
    """
    Calculate likelihood for the MC-dropout.
    The 'tau' constant will be calculated as the precision(variance),
    which is noted at the supplementray of the paper "Dropout as the Bayesian approximation; ICML 2016; Gal et al."

    Args:
        y (np.array[float]): Numpy array of ground truth values
        preds (np.array[float]): Numpy array of predictions
        sample_num (int, optional): Number of samples of differently predicted values. Defaults to 5.
        tau (float, optional): tau value of the MC-dropout (equation #7 of the original paper) 

    Returns:
        [float]: The likelihood. 
    """
    std = np.sqrt(std**2)# + 1/tau)
    pi = np.pi
    x1 = 0.5*np.log(2*pi) + np.log(std)
    x2 = 0.5*((y - preds)/std)**2

    return np.mean(x1 + x2)


def evaluate_model(Y, preds, std=[], mcdrop=False, **kwargs):
    """
    Evaluate model for various metrics.
    All keyword arguments will be passed for uncertainty evaluations.
    
    Args:
        Y(np.array) ground truth affinity values
        preds(np.array) predicted affinity values
        std(np.array or List) predicted standard derivation
        
    Return:
        result(dict) Dictionary including the evaluation results.
            keys: 'MSE', 'r2', 'CI', 'cal_error', 'cal_errors'
    """
    results = dict()
    
    results['MSE'] = str(np.mean((Y - preds)**2))
    results['r2'] = str(r_squared_error(Y, preds))
    results['CI'] = str(concordance_index(Y, preds))
    if len(std) != 0:
        results['cal_error'], cal_errors, _ = eval_uncertainty(preds, std, Y, **kwargs)
        if mcdrop:
            results['LL'] = str(-mcdrop_nll(Y, preds, std))
        else:
            results['LL'] = log_likelihood(Y, preds, std, **kwargs)
        results['ECE'] = str(np.mean(np.sqrt(cal_errors)))
        results['cal_error'] = str(results['cal_error'])
        results['OE'] = str(np.mean(eval_uncertainty(preds, std, Y, overconferr=True, **kwargs)[0]))
    results['AUPR'] = str(aupr(Y, preds, **kwargs))
    
    return results


def calibaration_eval(Y, preds, std):
    eval_uncertainty(preds, std, Y)
##########################################################################
import string
from itertools import cycle
from six.moves import zip

def label_axes(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (.9, .9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)
        
        
def fit_gamma_std(std):
    """
    Fit gamma function for given standard deviation.
    
    Args:
        std(np.array): A numpy array contains the standard deviation.
        
    Return:
        alpha, loc, beta
    """
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(std)
    return fit_alpha, fit_loc, fit_beta


def get_p_values(std, alpha, loc, beta, tensor=True):
    """
    Get p-values for std, under the gamma distribution parameterized with
    (alpha, loc, beta)

    Args:
        std(np.array): A numpy array contains the standard deviation.

    Return:
        np.array: A numpy array includes p-values for given data

    """
    return torch.Tensor(gamma.sf(std, alpha, loc, beta))


def to_one_hot_vector(num_class, label):
    b = np.zeros((label.shape[0], num_class))
    b[np.arange(label.shape[0]), label] = 1
    return b


def mixup_data(input_batch_drug, input_batch_protein, output_batch, alpha=0.1, prot_classes=26, drug_classes=65):
    batch_size = input_batch_drug.size()[0]
    index = torch.randperm(batch_size)
    input_batch_drug = torch.nn.functional.one_hot(input_batch_drug, drug_classes)
    input_batch_protein = torch.nn.functional.one_hot(input_batch_protein, prot_classes)

    mixed_drug = alpha*input_batch_drug + (1-alpha)*input_batch_drug[index,:]
    mixed_protein = alpha*input_batch_protein + (1-alpha)*input_batch_protein[index,:]
    mixed_y = alpha*output_batch + (1-alpha)*output_batch[index]

    return mixed_drug, mixed_protein, mixed_y
