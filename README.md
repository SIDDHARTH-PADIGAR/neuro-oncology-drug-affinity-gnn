# Drug-Target Interaction Prediction for Brain Cancer Therapeutics Using Graph Neural Networks

## Overview

This project develops a novel dual graph convolutional neural network (GCN) framework to predict drug-target binding affinities for brain cancer therapeutics. The framework leverages graph representations of both drug molecules and target proteins to capture complex molecular interactions, accelerating drug discovery in neuro-oncology.

## Features

- Implements advanced molecular graph learning using PyTorch Geometric and RDKit for drug and protein encoding.
- Curates high-quality datasets from PubChem, UniProt, Davis, and FDA drug data, enabling screening of over 170 drug candidates against more than 30 brain cancer-related protein targets.
- Achieves state-of-the-art predictive performance with RMSE below 0.26 and R squared above 0.8 on benchmark datasets.
- Incorporates explainability techniques including GNNExplainer for atom-level and residue-level interpretation of model predictions to improve interpretability and trust.
- Provides an end-to-end pipeline covering data preprocessing, graph construction, model training, hyperparameter optimization, evaluation, and visualization.
- Includes reproducible codebase and detailed documentation aimed at facilitating further research and potential journal publication.

## Workflow Architecture

The following diagram illustrates the end-to-end workflow of the drug-target interaction (DTI) prediction pipeline used in this project.

![Workflow Diagram](BC_DTI_WORKFLOW.drawio.jpg)

### Description

- **Data Acquisition:** Drug molecular information (SMILES strings) is collected from PubChem while protein sequences are sourced from UniProt.
- **Preprocessing Layer:** 
  - Drug SMILES strings are converted into molecular graphs using cheminformatics tools.
  - Protein sequences are transformed into residue graphs capturing neighboring contacts and sequence relationships.
- **Graph Encoding:** 
  - Molecular graphs are processed by a drug GCN encoder.
  - Residue graphs are processed by a protein GCN encoder.
- **Fusion and Regression:** The outputs (embeddings) from both encoders are concatenated and passed through a dense regression layer.
- **Model Training:** The architecture is trained end-to-end using regression loss (MSE), backpropagation, and hyperparameter tuning techniques with train, validation, and test splits.
- **Evaluation and Predictive Output:** Evaluates model predictions using metrics such as RMSE, R², and correlation coefficients, and outputs the final binding affinity score.

## Impact

This framework enables rapid in silico screening of candidate drugs, reducing traditional experimental time and resources by approximately 70 percent. It also provides novel insights into molecular mechanisms driving drug efficacy, supporting rational drug design and personalized therapeutic strategies for brain cancer.

## Getting Started

Instructions to install dependencies, prepare datasets, and run training and inference scripts are provided. Visualization notebooks demonstrate explainability outputs and molecular graphs.

## Future Work

Possible extensions include integration with multi-omics data, transfer learning on related cancer types, and deployment as an AI-assisted drug discovery platform.
