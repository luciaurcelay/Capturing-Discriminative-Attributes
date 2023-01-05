#! /bin/bash

BASEDIR=./

# Experiments

printf "\nRunning experiment 1\n"
python train.py -r True -e glove -ge glove.6B.50d -ed 50 -nw False -s 42 -m SVC -k rbf -c 1.0 -g scale -l1 False -cos False

# printf "\nRunning experiment 2\n"
