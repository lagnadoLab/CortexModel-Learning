# CortexModel-Learning

## Overview

A rate-based circuit model of layer 2/3 mouse V1 incorporating excitatory and inhibitory neuron populations to investigate adaptation during simple forms of learning described in [Reference placeholder]. This repository accompanies the manuscript and provides both a reference implementation of the model and the parameter optimization pipeline used in the study.

## Model Lineage

This repository builds upon the four-population rate-based model of layer 2/3 V1 introduced in [CortexModel](./https://github.com/username/CortexModel-Adaptation/releases/tag/v1.0.0) (release v1.0.0) and published in [Hinojosa, Kosiachkin *et al.* 2025](https://www.biorxiv.org/content/10.1101/2025.07.24.666602v2). 

Relative to the original implementation, the present model:
- Introduces an additional VIP subpopulation (VIP_Neg)
- Expands the circuit from four to five neuronal populations
- Updates connectivity and fitting procedures accordingly

## Repo Contents
This repository is organized into two complementary components:
- The **model implementation**, used in the Jupyter notebooks at the root level, which illustrates how specific parameter choices determine neural responses and their dynamics.
- The **fitting pipeline**, located in the `Fitting/` directory, which was used to identify the parameter sets that reproduce experimental data accurately.

### Model implementation (Jupyter Notebooks)
  
  - [Model_V1_Hab_Sess1.ipynb](./Model_V1_Hab_Sess1.ipynb) - Model simulation using fitted connection weights to reproduce average traces on the dataset of session 1 in the habituated group.

  - [Model_V1_Rew_Sess1.ipynb](./Model_V1_Rew_Sess1.ipynb) - Model simulation using fitted connection weights to reproduce average traces on the dataset of session 1 in the rewarded group.
  
 - [Model_V1_Hab_Sess6.ipynb](./Model_V1_Hab_Sess6.ipynb) - Model simulation using fitted connection weights to reproduce average traces on the dataset of session 6 in the habituated group.

  - [Model_V1_Rew_Sess6.ipynb](./Model_V1_Rew_Sess6.ipynb) - Model simulation using fitted connection weights to reproduce average traces on the dataset of session 6 in the rewarded group.
    
  - [Experimental data](./Experimental_data) - Experimental data used for both model implementation and fitting.

The notebooks run deterministic simulations using previously fitted parameters. They do not perform optimization.

### Fitting Pipeline

  - [Fitting](./Fitting) - directory containing the optimization pipeline used to identify parameter sets that reproduce the experimental data with high-performance computing (Artemis).

## System Requirements

### Hardware Requirements
**Model implementation** requires only a standard computer with enough CPU performance. 

The runtime of a single simulation is a few seconds on a standard desktop computer (16 GB RAM, 13th Gen Intel(R) Core(TM) i7-1360P 2.20 GHz).

The **fitting pipeline** was executed on the Artemis high-performance computing cluster. The runtime depends on the number of function evaluations (max_nfev), the number of initial conditions used in the optimization, and the number of CPU cores allocated. See the Fitting/ README for details.

### OS Requirements

The model was developed and tested under the Windows but should run on macOS and Linux.

## Installation Guide
The model was developed using Python 3.11.5 | packaged by Anaconda, Inc. 
The full virtual environment requirements are in [requirements.txt](./requirements.txt).
The main dependencies that should be installed/updated are:
```
ipykernel==6.28.0
ipython==8.25.0
ipywidgets==8.1.3
lmfit==1.3.2
matplotlib==3.9.1.post1
matplotlib-inline==0.1.6
numdifftools==0.9.41
numpy==2.0.1
pandas==2.2.2
scikit-learn==1.5.1
scipy==1.14.0
seaborn==0.13.2
```

To install the **model implementation** and work with the Jupyter Notebook files download and unzip the main folder of this GitHub repository.
<p><b>Important:</b> For all data uploads in the Jupyter notebook files to work correctly, please keep pathways and hierarchy of files and folders unchanged.</p>
<p>For working with different datasets, provide specific file pathway to data files in data upload section. Data should be a one-dimensional .txt file with one data point per row. **Important:** a frame interval of 0.164745 s (correspoding to a 6.07 Hz acquisition rate) was used in our work. Please adjust the frame interval according to your recording framerate.</p>

For instructions on how to run the **fitting pipeline** using Artemis, check the Readme file in the /Fitting directory.

## Results
We fitted the model to the response dynamics of Pyramidal cells (PCs) and three types of interneurons: PVs, SSTs and two subsets of VIPs responding to the stimulus either with increased amplitude (VIP_Pos) or being supressed by the stimulus (VIP_Neg). 
In [Model_V1_Hab_Sess1.ipynb](./Model_V1_Hab_Sess1.ipynb) and [Model_V1_Rew_Sess1.ipynb](./Model_V1_Rew_Sess1.ipynb) the model was fitted to average response traces from mice exposed to a novel visual stimulus, either presented alone (habituation group) or paired with reward (reward group). In [Model_V1_Hab_Sess6.ipynb](./Model_V1_Hab_Sess6.ipynb) and [Model_V1_Rew_Sess6.ipynb](./Model_V1_Rew_Sess6.ipynb) the model was fitted to average traces from mice that were familiar with the stimulus, either habituated to it or trained to associate it with reward.

## Citation

For usage of the model and associated manuscript please cite [Reference placeholder].