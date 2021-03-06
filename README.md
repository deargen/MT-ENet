# **Improving evidential deep learning via multi task learning**

It is a repository of AAAI2022 paper, “*[Improving evidential deep learning via multi-task learning](https://arxiv.org/abs/2112.09368)*”, by Dongpin Oh and Bonggun Shin.

This repository contains the code to reproduce the Multi-task evidential neural network (MT-ENet), which uses the Lipschitz MSE loss function as the additional loss function of the [evidential regression network](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html) (ENet). The Lipschitz MSE loss function can improve the accuracy of the ENet while preserving its uncertainty estimation capability, by avoiding gradient conflict with the NLL loss function—the original loss function of the ENet.

<p align="center">
<img src="https://github.com/deargen/MT-ENet/blob/main/pic/synthetic_experiment.png" alt="drawing" width="700"/>
</p>

## **Setup**

Please refer to "requirements.txt" for requring packages of this repo.

```bash
pip install -r requirements.txt
```


## Training the ENet with the Lipschitz-MSE loss: example

```python
from mtevi.mtevi import EvidentialMarginalLikelihood, EvidenceRegularizer, modified_mse
...
net = EvidentialNetwork() ## Evidential regression network
nll_loss = EvidentialMarginalLikelihood() ## original loss, NLL loss
reg = EvidenceRegularizer() ## evidential regularizer
mmse_loss = modified_mse ## lipschitz MSE loss
...
for inputs, labels in dataloader:
	gamma, nu, alpha, beta = net(inputs)
	loss = nll_loss(gamma, nu, alpha, beta, labels)
	loss += reg(gamma, nu, alpha, beta, labels)
	loss += mmse_loss(gamma, nu, alpha, beta, labels)
	loss.backward()	
```

## **Quick start**

- **Synthetic data experiment.**

```bash
python synthetic_exp.py
```

- **UCI regression benchmark experiments.**

```bash
python uci_exp_norm -p energy
```

- **Drug target affinity (DTA) regression task on KIBA and Davis datasets.**

```bash
python train_evinet.py -o test --type davis -f 0 --evi # ENet
python train_evinet.py -o test --type davis -f 0  # MT-ENet
```

- **Gradient conflict experiment on the DTA benchmarks**

```bash
python check_conflict.py --type davis -f 0 # Conflict between the Lipschitz MSE (proposed) and NLL loss. 
python check_conflict.py --type davis -f 0 --abl # Conflict between the simple MSE loss and NLL loss.
```

## Characteristic of the Lipschitz MSE loss

<p align="center">
<img src="https://github.com/deargen/MT-ENet/blob/main/pic/lipschitzMSE.png" alt="drawing" width="700"/>
</p>

- The Lipschitz MSE loss function can support training the ENet to more accurately predicts target values.
- It regularizes its gradient to prevent gradient conflict with the NLL loss--the original loss function--if the NLL loss increases predictive uncertainty of the ENet. 
- Please check our [paper](https://arxiv.org/abs/2112.09368) for details.
