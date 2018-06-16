# DeepVS

This is the PyTorch implementation of the DeepVS neural network architecture, which is describe in the paper [Boosting Docking-Based Virtual Screening with Deep Learning](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.6b00355). 

DeepVS is a deep learning approach to improve the identification of active ligands in docking-based virtual screening. DeepVS uses the output of a docking program and learns how to extract relevant features from basic data such as atom and residues types obtained from protein−ligand complexes.

## Running the experiment

The [train.py
](train.py) file implements the cross-validation experiment reported in the paper. The code should be intuitive. You can run it as follows:

```
python train.py
```

Note that in this version of the code we use ReLU and add dropout. These changes made our architeture more robust.

## DUD preprocessed data

In order to run the code, you will need our preprocessed vina ouput [data](https://www.dropbox.com/s/4486qf7lpsxuwy2/dud_vinaout_deepvs.zip?dl=0).

After downloading the data, unzip it and put the folder dud_vinaout_deepvs in the same directory as [train.py
](train.py)
  
## Prerequisites

Python 2.7 
Pytorch 0.2.0_4

### Paper Reference

If this code is useful for you somehow please cite our paper:

```
@article{doi:10.1021/acs.jcim.6b00355,
author = {Pereira, Janaina Cruz and Caffarena, Ernesto Raúl and dos Santos, Cicero Nogueira},
title = {Boosting Docking-Based Virtual Screening with Deep Learning},
journal = {Journal of Chemical Information and Modeling},
volume = {56},
number = {12},
pages = {2495-2506},
year = {2016},
doi = {10.1021/acs.jcim.6b00355},
note ={PMID: 28024405},
URL = {https://doi.org/10.1021/acs.jcim.6b00355},
eprint = {https://doi.org/10.1021/acs.jcim.6b00355}
}
```
## License

This project is licensed under the Apache License v2.0 - see the [LICENSE.md](LICENSE.md) file for details

