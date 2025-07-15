# Sequential memory based on predictive coding for auditory memory

## 1. Description
This repository contains code to perform experiments with **closed-loop temporal predictive coding** in auditory memory tasks, which is discussed in the IJCNN 2025 paper [A General Closed-loop Predictive Coding Framework for Auditory Working Memory]([https://proceedings.neurips.cc/paper_files/paper/2023/hash/8a8b9c7f979e8819a7986b3ef825c08a-Abstract-Conference.html](https://arxiv.org/abs/2503.12506)).

## 2. Installation
1. `conda env create -f environment.yml`  
2. `conda activate auditorymem`
3. `pip install -e .`

## 3. Use
You can easily run the model with:

python auditory_seq_memory.py

This command is used to go through the whole ESC 50 dataset. It can be easily transfer to other datasets.
