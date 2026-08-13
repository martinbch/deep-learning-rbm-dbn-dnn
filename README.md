# Deep Learning with RBMs, DBNs and DNNs

This repository contains a deep learning project focused on Restricted
Boltzmann Machines (RBMs), Deep Belief Networks (DBNs) and Deep Neural
Networks (DNNs).

The project studies unsupervised representation learning and image generation
with Binary AlphaDigits, as well as supervised classification with MNIST. It
also compares randomly initialized networks with networks pre-trained layer by
layer using RBMs.

## Project objectives

The project is organized around the following objectives:

1. Implement and train an RBM from scratch.
2. Build a DBN by stacking RBMs and generate new Binary AlphaDigits images.
3. Implement a DNN for MNIST classification.
4. Compare standard supervised initialization with RBM/DBN pre-training.
5. Study the influence of architecture, hidden-layer size and training-set
   size on model performance.

## Methods

- Restricted Boltzmann Machine (RBM)
- Deep Belief Network (DBN)
- Deep Neural Network (DNN)
- Gibbs sampling for image generation
- Layer-wise unsupervised pre-training
- Supervised fine-tuning with backpropagation
- Softmax classification

## Repository contents

- `P4_AlphaDigit.ipynb`: RBM and DBN experiments on Binary AlphaDigits,
  including image generation and hyperparameter analysis.
- `P5_MNIST.ipynb`: MNIST classification experiments and comparison of
  randomly initialized and pre-trained DNNs.
- `principal_RBM_alpha.py`: RBM implementation and training utilities.
- `principal_DBN_alpha.py`: DBN implementation based on stacked RBMs.
- `principal_DNN_MNIST.py`: DNN implementation, pre-training and supervised
  fine-tuning for MNIST.
- `deep-learning-rbm-dbn-dnn-report.pdf`: project report.

## Data

The datasets are not included in this repository. The notebooks require the
Binary AlphaDigits data and the binary MNIST files to be available locally.
Dataset paths must be configured before running the experiments.

## Setup and dependencies

Python 3.10 or later is recommended. The main dependencies are:

```text
numpy
scipy
matplotlib
tqdm
jupyter
```

Install the dependencies with:

```bash
pip install numpy scipy matplotlib tqdm jupyter
```

## Running the project

From the repository root, start Jupyter:

```bash
jupyter notebook
```

Then open `P4_AlphaDigit.ipynb` or `P5_MNIST.ipynb` and run the cells in
order after configuring the dataset paths.

## Authors

- Martin Boucher
- Thomas Balsalobre

Repository: `deep-learning-rbm-dbn-dnn`