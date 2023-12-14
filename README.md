# Heterogeneous GNN for Ligand-Binding Protein Design
CS224W course project  

Authors: Braxton Bell, Tianyu Lu

## Inference

<a href="https://colab.research.google.com/drive/1x_-gh5zWBCluOha4Z-WrfdzF6EFOt2IB?usp=sharing"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><br />

The simplest way to get started is with our Colab notebook. All user options can
be specified in the input form in the notebook.

## Training

To train the model on the full MISATO dataset, a few steps are required:
1. Download the full MISATO datasets and train/val/test splits [here](https://zenodo.org/records/7711953) and move them to `../data/MD/h5_files/` for MD trajectories and `../data/MD/splits` for the `.txt` train/val/test splits.
2. Create a conda environment with PyTorch, PyTorch Geometric, and its associated packages (see `env.yml`)
3. For training to converge in a reasonable amount of time, ensure access to a GPU
4. See train.py for the main training script and options (e.g. model hyperparameters)

## Model Description

For a detailed description of the model architecture, performance, and discussion, see this [Medium blog post](https://medium.com/@tianyulu710/heterogeneous-graph-neural-network-for-ligand-binding-protein-design-1d65f7a55c95)
