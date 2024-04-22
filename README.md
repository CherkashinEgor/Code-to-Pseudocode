# Code-to-Pseudocode Translation Using Transformer Architecture

## Overview
This repository is part of a project for CSCI 499. It focuses on testing various architectures for translating programming code into pseudocode. Specifically, this repo is dedicated to the implementation and evaluation of the transformer architecture. 

## Datasets
This project utilizes two datasets:
- **Django Dataset**: A dataset for code synthesis from [Django](https://github.com/odashi/ase15-django-dataset)
- **CoNaLa Dataset**: The Code/Natural Language Challenge dataset. It can be found [here](https://conala-corpus.github.io/).

## Project Structure
- `data.py`: Handles data loading and preparation.
- `model.py`: Contains the definition of the Transformer model.
- `train.py`: Scripts to train the model using the datasets prepared in `data.py`.
- `translate.py`: Contains functions for translating code to pseudocode using the trained model.
- `util.py`: Includes various utility functions such as mask generation and evaluation metrics.
- `tokenizer.py`: Manages tokenization processes.
- `main.py`: The entry point for training the models.
