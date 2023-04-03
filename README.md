# shapelet_transformer
This is the code of "An Adversarial Robust Behavior Sequence Anomaly Detection Approach Based on Critical Behavior Unit Learning", which is submitted to IEEE Transactions of Computers.

The project mainly consists of the following components:
- pre-log: This directory contains the implementation for preprocessing the DroidEvolver dataset.
- shapelet_generation: This directory is used to extract the shapelets of sequences from the input data.
- Lcs-transform: This directory contains the implementation for the long common subsequence (LCS) algorithm to extract behavior units from the perturbed behavior sequence
- em_add__transformer_train.py: This is the main code for implementing Feature Extraction and Behavior Classification
- defense-gan: The code in this directory contains the implementation of Defense-GAN. 
