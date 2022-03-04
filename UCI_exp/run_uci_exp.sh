mkdir log

nohup python -u uci_exp_norm.py -p boston > log/boston.log &
nohup python -u uci_exp_norm.py -p yacht > log/yacht.log &
nohup python -u uci_exp_norm.py -p energy > log/energy.log &
wait
nohup python -u uci_exp_norm.py -p concrete > log/concrete.log &
nohup python -u uci_exp_norm.py -p wine > log/wine.log &
wait
nohup python -u uci_exp_norm.py -p kin8nm > log/kin8nm.log &
nohup python -u uci_exp_norm.py -p power > log/power.log &
wait
nohup python -u uci_exp_norm.py -p protein > log/protein.log 
nohup python -u uci_exp_norm.py -p navel > log/navel.log 
wait
