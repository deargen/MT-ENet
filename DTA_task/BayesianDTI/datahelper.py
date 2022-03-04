"""
Original source of datahelper.py is from the DeepDTA repository.

This source includes pre-processing for drug target interaction tasks, including
the inherted classes from torch.utils.data.Dataset.

See the "Load data" parts of DTI experiment notebooks.

"""
import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
from abc import ABC, abstractmethod
import random
import copy
from BayesianDTI.predictor import EviNetDTIPredictor

## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ## 

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
                "U": 19, "T": 20, "W": 21, 
                "V": 22, "Y": 23, "X": 24, 
                "Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
             ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
             "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
             "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
             "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
             "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
             "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
             "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
             "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
             "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
             "t": 61, "y": 62, 'MASK': 63} ### Canonical SMILES

CHARCANSMILEN = 62 # +1 for mask

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}#,
                #'MASK': 65, 'UNK': 66, 'START': 67, 'END': 68, 'REP': 69} ### Iso SMILES

CHARISOSMILEN = 64 # +2 for mask and UNK.


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ## 

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch]-1)] = 1 

    return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind))) 
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch])-1] = 1

    return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN) ## Masked token is 0.
    for i, ch in enumerate(line[:MAX_SMI_LEN]): #   x, smi_ch_ind, y
        try:
            X[i] = smi_ch_ind[ch]
        except KeyError: # UNK
            X[i] = smi_ch_ind['C']
    return X #.tolist()

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN) ## Masked token is 0
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        try:
            X[i] = smi_ch_ind[ch]
        except:
            X[i] = 0
    return X #.tolist()


## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
    """AI is creating summary for DataSet

    Args:
        object ([type]): [description]
    """    
    def __init__(self, fpath, seqlen, smilen, need_shuffle = False):
        """AI is creating summary for __init__

        Args:
            fpath ([type]): [description]
            seqlen ([type]): [description]
            smilen ([type]): [description]
            need_shuffle (bool, optional): [description]. Defaults to False.
        """        
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        #self.NCLASSES = n_classes
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
        self.charsmiset_size = CHARISOSMILEN


    def read_sets(self, FLAGS): ### fpath should be the dataset folder /kiba/ or /davis/
        fpath = FLAGS.dataset_path
        setting_no = FLAGS.problem_type
        print("Reading %s start" % fpath)

        test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
        train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))

        return test_fold, train_folds

    def parse_data(self, FLAGS,  with_label=True): 
        fpath = FLAGS.dataset_path  
        print("Read %s start" % fpath)

        ligands = json.load(open(fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath+"proteins.txt"), object_pairs_hook=OrderedDict)

        Y = pickle.load(open(fpath + "Y","rb"), encoding='latin1') ### TODO: read from raw
        if FLAGS.is_log:
            Y = -(np.log10(Y/(math.pow(10,9))))

        XD = []
        XT = []

        if with_label:
            for d in ligands.keys():
                XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
        else:
            for d in ligands.keys():
                XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))

        return XD, XT, Y

    
class DataSetting:
    def __init__(self):
        self.dataset_path = 'data/davis'
        self.problem_type = '1'
        self.is_log = True

    
def load_data():
    pass


def parse_molecule(fpath, smi_len=100, smi_dict=None, f_format='json'):
    """
    Parse the SMILE molecule file.
    
    Args:
        fpath(str): The file path. The file format should be json or text file including
            a one smile molecule for each line.
        
        Or,
        
        fpath(List): You can import the list containing SMILES itself.
        
        smi_dict(Dict): A python dictionary represents mapping between SMILES letters and
            integers.
    """
    if smi_len == None:
        smi_len = CHARISOSMILEN
    if smi_dict == None:
        smi_dict = CHARISOSMISET
    
    if f_format == 'json':
        ligands = json.load(open(fpath), object_pairs_hook=OrderedDict)
    elif f_format == 'txt':
        ligands_raw = open(fpath).readlines()
        ligands = OrderedDict(zip(range(len(ligands_raw)), ligands_raw))
    elif type(fpath) == list:
        ligands_raw = fpath
        ligands = OrderedDict(zip(range(len(ligands_raw)), ligands_raw))
        
    ligands_list = []
    for d in ligands.keys():
        ligands_list.append(label_smiles(ligands[d], smi_len, smi_dict))
    return ligands_list


def parse_protein(fpath, prot_len=1000, prot_dict=None, f_format='json'):
    """
    Parse the protein file.
    
    Args:
        fpath(str): The file path. The file format should be json or text file including
        a one protein for each line.
        
        Or,
        
        fpath(List): You can import the list containing residues itself.
        
        prot_dict(Dict): A python dictionary represents mapping between protein residues and
            integers.
    
    """
    if prot_len == None:
        prot_len = CHARPROTLEN
    if prot_dict == None:
        prot_dict = CHARPROTSET
    
    if f_format == 'json':
        proteins = json.load(open(fpath), object_pairs_hook=OrderedDict)
    elif f_format == 'txt':
        proteins_raw = open(fpath).readlines()
        proteins = OrderedDict(zip(range(len(proteins_raw)), proteins_raw))
    elif type(fpath) == list:
        proteins_raw = fpath
        proteins = OrderedDict(zip(range(len(proteins_raw)), proteins_raw))
        
    proteins_list = []
    for p in proteins.keys():
        proteins_list.append(label_sequence(proteins[p].upper(), prot_len, prot_dict))
    return proteins_list

    
class DTIDataset(Dataset):
    def __init__(self, smiles, proteins, Y, drug_idx, protein_idx):
        """
        Pytorch Dataset for DTI dataset.
        
        Args:
            smiles(List[np.array]) A List of numpy array consists of converted smile strings to
                integers using parse_molecule() or DataSet.parse_data()
            
            proteins(List[np.array]) A List of numpy array consists of converted protein strings to
                integers using parse_protein() or DataSet.parse_data()
                
            Y(np.array) A numpy array represents the true affinity value.
                
                e.g ) Y[i,j] represents the affinity value of i`th drug and j`th protein.
                    i`th drug: smiles[i]; j`th protein: proteins[j]
                    
            drug_idx(List) A List of smile index for "smiles", "Y".
            
            protein_idx(List) A List of protein index for "proteins", "Y".
        """
        
        self.X = []
        for d_i, p_i in zip(drug_idx, protein_idx):
            self.X.append((smiles[d_i], proteins[p_i], Y[d_i,p_i]))
        self.X = np.array(self.X) 
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, key):
        if type(key) == slice:
            self.X = self.X[key]
            return self
        return self.X[key]
    
class DTIDataset_PDB(Dataset):
    def __init__(self, smiles, proteins, Y=[]):
        """
        Pytorch Dataset for DTI dataset, using Protein-Ligand complex dataset of PDB.
        
        Args:
            smiles(List[np.array]) A List of numpy array consists of converted smile strings to
                integers using parse_molecule() or DataSet.parse_data()
            
            proteins(List[np.array]) A List of numpy array consists of converted protein strings to
                integers using parse_protein() or DataSet.parse_data()
        
            Y(np.array) A numpy array represents the true affinity value.
        """
        self.X = []
        for i in range(len(smiles)):
            if len(Y) == 0:
                self.X.append((smiles[i], proteins[i]))
            else:
                self.X.append((smiles[i], proteins[i], Y[i]))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, key):
        if type(key) == slice:
            return DTIDataset(self.X[key])
        return self.X[key]
            

class DTIDatasetPrediction(Dataset):
    def __init__(self, smiles, proteins, idx_pairs=None):
        """
        Pytorch Dataset for DTI dataset.
        
        Args:
            smiles(List[np.array]):
                A List of SMILES or np.array
                
            proteins(List[np.array]):
                A List of Protein seq or np.array
                
            idx_pairs(Tuple(smiles_idx, proteins_idx)):
                Target index pairs to predict,
                If it is None, generates all possible pairs.
                
        """
        
        self.X = []
        self.idx_pairs = idx_pairs
        
        if idx_pairs == None:
            self.idx_pairs = []
            for i in range(len(smiles)):
                for j in range(len(proteins)):
                    self.idx_pairs.append((i,j))
        
        for d_i, p_i in self.idx_pairs:
            self.X.append((smiles[d_i], proteins[p_i], 0.0)) #0.0 is dummy.
        ### TODO: Do something

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, key):
        if type(key) == slice:
            return DTIDatasetPrediction(self.X[key])
        return self.X[key]

    
def collate_dataset(batch, test=False):
    """
    Preprocessing for given batch.
    It will use for the Torch DataLoader(collate_fn=collate_dataset).
    
    Args: 
        batch(Torch.Tensor)
    
    return:
        List(Torch.Tensor)
    """
    drug_batch = []
    protein_batch = []
    y_batch = []

    for data in batch:
        drug_batch.append(data[0])
        protein_batch.append(data[1])
        if not test:
            y_batch.append(data[2])
    if not test:
        return [torch.LongTensor(drug_batch), torch.LongTensor(protein_batch), torch.FloatTensor(y_batch)]
    else:
        return [torch.LongTensor(drug_batch), torch.LongTensor(protein_batch)]
        

def collate_uncertainty_dataset(batch, Y=False):
    """
    Preprocessing for given batch.
    It will use for the Torch DataLoader(collate_fn=collate_dataset).
    
    Args: 
        batch(Torch.Tensor)
        Y(bool) : use true label or not.
    
    return:
        List(Torch.Tensor)
    """
    drug_batch = []
    protein_batch = []
    mu_batch = []
    std_batch = []
    
    ## Mannually decide Y.
    if len(batch[0]) == 5:
        Y = True
    
    if Y:
        y_batch = []
    for d, p, mu, std, y in batch:
        drug_batch.append(d)
        protein_batch.append(p)
        mu_batch.append(mu)
        std_batch.append(std)
        if Y:
            y_batch.append(y)
    
    if Y:
        return [torch.LongTensor(drug_batch), torch.LongTensor(protein_batch),
            torch.FloatTensor(mu_batch), torch.FloatTensor(std_batch), torch.FloatTensor(y_batch)]
    else:
        return [torch.LongTensor(drug_batch), torch.LongTensor(protein_batch),
            torch.FloatTensor(mu_batch), torch.FloatTensor(std_batch)]


def get_fold(data_type, FOLD_NUM):
    data_type = data_type.lower()
    
    if not (data_type in ['davis','kiba']):
        print("wrong data type {}").format(data_type)
    if not (FOLD_NUM in [0,1,2,3,4]):
        print("wrong fold number {}").format(FOLD_NUM)
        
    class DataSetting:
        def __init__(self):
            self.dataset_path = 'data/{}/'.format(data_type)
            self.problem_type = '1'
            self.is_log = True if data_type == 'davis' else False

    data_setting = DataSetting()

    dataset = DataSet(data_setting.dataset_path,
                      1200 if data_type == 'davis' else 1000,
                      85 if data_type == 'kiba' else 100) ## KIBA (1000,100) DAVIS (1200, 85)
    smiles, proteins, Y = dataset.parse_data(data_setting)
    test_fold, train_folds = dataset.read_sets(data_setting)

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
    
    fold = dict()
    
    fold['test_drug'] = label_row_inds[test_fold]
    fold['test_protein'] = label_col_inds[test_fold]

    train_fold_sum = []
    for i in range(5):
        if i != FOLD_NUM:
            train_fold_sum += train_folds[i]

    fold['train_drug'] = label_row_inds[train_fold_sum]
    fold['train_protein'] = label_col_inds[train_fold_sum]

    fold['valid_drug'] = label_row_inds[train_folds[FOLD_NUM]]
    fold['valid_protein'] = label_col_inds[train_folds[FOLD_NUM]]
    
    fold['smiles'] = smiles
    fold['proteins'] = proteins
    fold['Y'] = Y
    
    return fold


def pick_random_idx(idices, num, seed=0):    
    """
    Pop the "num" number of random samples from the sequence "indices"

    Args:
        idices (Sequence): 
        num (Int): The number to pop up the samples from the "idices" 
        seed(Int): Random seed
    Returns:
        remaining_sequences, pop-upped sequences 
    """
    shuffled_idx = copy.deepcopy(idices)
    random.Random(seed).shuffle(shuffled_idx)
    picked_idx = shuffled_idx[:num]
    remain_idx = shuffled_idx[num:]
    return remain_idx, picked_idx

def pick_uncertain_idx(remain_idx, N, dataset, model, seed=None):
    """
    Get the idices of the training set, which have high uncertainties.

    Args:
        remain_idx (Sequence[Int]): The indicies of the training set to split by its uncertainty
        N (Int): Number of samples to pick up highly uncertain samples.
        dataset (BayesianDTI.datahelper.Dataset): Dataset class
        model (torch.nn.Module): The evidential network to estimate the uncertainty.

    Returns:
        (remaining_indices, picked indices)
        picked_indices contains the indices of the "dataset"
        , which represent top "N_init" uncertain samples.
        And the ramining_indices is the remaining indices after excluding
        uncertain samples.
    """
    idices = np.array(copy.deepcopy(remain_idx))
    dataloader = DataLoader(dataset[idices], batch_size=256, collate_fn=collate_dataset)
    model.eval()
    predictor = PriorNetDTIPredictor()
    _, std, _, _ = predictor(dataloader, model)
    total_std = std['epistemic'] #+ std['aleatoric']
    uncertainty_idices = total_std.argsort()[::-1][:N]
    picked_idices = idices[uncertainty_idices]
    remaining_idices = np.setdiff1d(idices, picked_idices, assume_unique=True)
    return remaining_idices, picked_idices

def split_integer(N, K):
    """
    Split Integer to K number of smaller integer.
    Args:
        N (Int): 
        K (Int): 
    Returns:
        [List]: A List containing splited integers.
    """
    a = N//K
    diff = N - a*K
    N_samples = []
    for i in range(K):
        if diff > 0:
            N_samples.append(a + 1)
        else:
            N_samples.append(a)
        diff = diff - 1
    return N_samples

import random
def generate_random_residues(n, l = 1000, seed=None):
    residues = []
    for i in range(n):
        if seed != None:
            random.seed(seed+i)
        length = random.randint(1,l)
        residues.append(''.join(random.choices(list(CHARPROTSET.keys()), k=length)))
    return residues


def generate_random_smiles(n, l = 100):
    smiles = []
    for i in range(n):
        smiles.append(''.join(random.choices(list(CHARISOSMISET.keys()), k=l)))
    return smiles 

def fasta2dict(fil, uniprot=True, pass_seq=None):
    """
    Read fasta-format file fil, return dict of form scaffold:sequence.
    Note: Uses only the unique identifier of each sequence, rather than the 
    entire header, for dict keys. 
    """
    dic = {}
    cur_scaf = ''
    cur_seq = []
    pass_line = False
    for line in open(fil):
        if pass_seq != None and line.startswith(">"):
            print(line)
            pass_line = False
            if type(pass_seq) == list:
                for p_seq_i in pass_seq:
                    if p_seq_i in line:
                        pass_line = True
                        continue
            elif type(pass_seq) == str:
                if pass_seq in line:
                    pass_line = True
        
        if pass_line:
            continue

        if line.startswith(">") and cur_scaf == '':
            cur_scaf = line.split(' ')[0].split('|')[1]
        elif line.startswith(">") and cur_scaf != '':
            dic[cur_scaf] = ''.join(cur_seq)
            cur_scaf = line.split(' ')[0].split('|')[1]
            cur_seq = []
        else:
            cur_seq.append(line.rstrip())
    dic[cur_scaf] = ''.join(cur_seq)
    return dic
