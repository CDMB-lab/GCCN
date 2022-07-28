# GCCN
source code of **G**raph **C**apsule **C**onvolutional **N**etwork (GCCN). <br>
The data used in this study can be openly available at http://adni.loni.usc.edu/.

## Dependencies
The script has been tested running under Python 3.9.0, with the following packages installed (along with their dependencies): <br>

* numpy==1.21.2 <br>
* networkx==2.6.3 <br>
* torch==1.11.0 <br>
* torch-geometric==2.0.4 <br>
* In addition, CUDA 11.3 have been used on NVIDIA GeForce RTX 3080. <br>

## Overview
The repository is organised as follows: <br>
* `dataset.py`: contains the implementation of **H**eterogeneous **P**athogenic **I**nformation **A**ssociation **G**raphs (HPIAGs); <br>
* `transform.py`: include the implementation of batching operation and graph related feature engineering; <br>
* `disentangle.py`: contains a variety implementation of disentangling functions; <br>
* `denseconv.py`: is an simply implementation of `DenseGCNConv`; <br>
* `layers.py`: implements the three layers (primary layer & digital layer & reconstruction layer); <br>
* `models.py`: contains the implementation of the HGCCN; <br>
* `custom_function.py`: contains the HGCCN related operation; <br>
* `sparsemax.py`: contains the implementation of `Sparsemax`; <br>
* `parameters.py`: including all the parameters involoved in model; <br>
* `train_eval_helper.py`: contains the cross-validation related helper functions. <br>
* Finally, `main.py` puts all of the above together and be used to execute a full training run.
