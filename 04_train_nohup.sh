#!/bin/bash
nohup python main.py train --gpu=1 --xla=2 -ep=4 -bs=10 -lr=0.0005 > train-nohup.out 2>&1 &
