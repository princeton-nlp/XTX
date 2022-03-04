# XTX: eXploit - Then - eXplore

**Project page:** https://sites.google.com/princeton.edu/xtx

## Requirements
First clone this repo using `git clone https://github.com/princeton-nlp/XTX.git`

Please create two conda environments as follows:
1. `conda env create -f yml_envs/jericho-wt.yml`  
    a. `conda activate jericho-wt`  
    b. `pip install git+https://github.com/jens321/jericho.git@iclr` 
2. `conda env create -f yml_envs/jericho-no-wt.yml`

The first set of commands will create a conda environment called `jericho-wt` which has added actions to the game grammar for specific games (see games with * in the paper). The second command will create another conda environment called `jericho-no-wt` which installs an unmodified version of the Jericho library. 

## Training
All code can be run from the root folder of this project. Please follow the commands below for each specific model: 
- XTX: `sh scripts/run_xtx.sh`
- XTX (no-mix): `sh scripts/run_xtx_no_mix.sh`
- XTX (uniform): `sh scrtips/run_xtx_uniform.sh`
- XTX ($\lambda$ = 0, 0.5, or 1): `sh scripts/run_xtx_ablation.sh`
- INV DY: `sh scripts/run_inv_dy.sh` 
- DRRN: `sh scripts/run_drrn.sh`

### Notes
- You can use `analysis/sample_env.py` for quickly playing around with a sample Jericho environment. Run it using `python3 -m analysis.sample_env`.

- You can use `analysis/augment_wt.py` for generating the missing action candidates that can be added to the game grammar (games with * in the paper). Run it using `python3 -m analysis.augment_wt`. 

- Note that all models should finish within a day or two given 1 gpu and 8 cpus, except for games where Jericho's valid action handicap is slow (e.g. Library, Dragon). Since Jericho's valid action handicap heavily relies on parallelization, increasing the number of cpus also results in good speedups (e.g. 8 -> 16). 

## Acknowledgements
We used [Weights & Biases](https://wandb.ai/home) for experiment tracking and visualizations to develop insights for this paper.

Some of the code borrows from the [TDQN](https://github.com/microsoft/tdqn) repo. 

For any questions please contact Jens Tuyls (`jtuyls@princeton.edu`).
