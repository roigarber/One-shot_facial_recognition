
# One-Shot Facial Recognition

![Image](https://github.com/user-attachments/assets/0c612a40-8c28-4d26-96a8-27c254daf721)


This assignment uses convolutional neural networks (CNNs) to perform facial recognition via a one-shot learning approach. Based on the paper *“Siamese Neural Networks for One-shot Image Recognition”*, training is conducted on the LFW-a dataset with a predefined train/test split (no subject overlap), so that, given two previously unseen face images, the model must determine whether they depict the same person.



## Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Model architecture](#model-architecture)  
4. [Training and Experiments](#training-and-Experiments)    
5. [Conclusion](#conclusion)  
6. [Authors](#authors)  
## Overview


This project implements a Siamese-network-based one-shot learning model in PyTorch for face recognition. Various distance metrics and data-augmentation strategies are compared to maximize accuracy on the chosen dataset.

## Dataset

The “Labeled Faces in the Wild-a” (LFW-a) image collection is a database of labeled face photographs intended for studying face recognition in unconstrained settings. We trained our model on the provided LFW-a train and test splits, which contain 3,200 one-shot pairs in total.  

Here are examples of some of the aligned face pairs from the dataset:  

![Image](https://github.com/user-attachments/assets/192d8181-df9a-4193-8e39-5a3dde257591)

An analysis of the original splits shows 3,200 examples, of which 2,200 are used for training (1,100 matched pairs and 1,100 mismatched pairs) and 1,000 for testing (500 matched, 500 mismatched). To better monitor performance during training and guard against overfitting—as recommended in the reference paper—we further carved out 15 % of the training data as a validation set.  This validation set was used to tune hyperparameters and select the best model checkpoints before final evaluation.  

## Model architecture

The network architecture follows Koch et al.’s Siamese design almost exactly, with the only change being that all inputs are resized to 105×105 pixels (rather than the original 250×250) to accommodate compute limits. Each branch processes a single grayscale image through four convolutional blocks:

1. **Block 1:** 64 filters of size 10×10 → BatchNorm → ReLU → 2×2 max-pool  
2. **Block 2:** 128 filters of size 7×7 → BatchNorm → ReLU → 2×2 max-pool  
3. **Block 3:** 128 filters of size 4×4 → BatchNorm → ReLU → 2×2 max-pool  
4. **Block 4:** 256 filters of size 4×4 → BatchNorm → ReLU  

The 256×6×6 feature map is then flattened and passed through a fully connected layer of size 4,096 with sigmoid activation. Finally, the absolute difference between the two 4,096-dim vectors is fed to a single sigmoid output unit that predicts similarity.

![Image](https://github.com/user-attachments/assets/88d6e947-7669-4bb5-83d8-49d21e28e168)
  
## Training and Experiments

All experiments began with batch size 64, 10 epochs, and SGD vs. Adam optimizers at learning rates {1e-3, 1e-4}. Early stopping was included but proved unnecessary. Key steps:

- **Baseline:** SGD at 1e-3 vs. 1e-4, no momentum  
- **Momentum sweep:** added momentum 0.9 to SGD  
- **Optimizer comparison:** switched to Adam  
- **Dropout regularization:** inserted p = 0.25, then p = 0.15, finally p = 0.07 to combat overfitting  

below are three representative examples:

- **Model 1:** SGD optimizer, learning rate 0.001, batch size 64, 10 epochs.  
- **Model 2:** Adam optimizer learning rate = 0.001, batch size 64, 10 epochs.  
- **Model 3:** Adam optimizer, learning rate 0.0001, dropout p = 0.07, batch size 64, 10 epochs

**Model 1 graphs:**
![Image](https://github.com/user-attachments/assets/0e2feb5e-7d86-4a8d-bd32-ae2d985b009f)
**Model 2 graphs:**
![Image](https://github.com/user-attachments/assets/d41784a7-9e02-446f-abd4-80760da2e784)
**Model 3 graphs:**
![Image](https://github.com/user-attachments/assets/50cf2595-00e0-4ae7-b87f-19ac28354913)

The best configuration—Adam optimizer at lr = 1e-4, batch size 64, dropout p = 0.07—struck the optimal balance of low validation loss and high test accuracy.## Results

From all configurations, the third model (Adam at lr = 0.0001 with dropout 0.07) delivered the best balance of high accuracy and low overfitting. The held-out test accuracy reached **0.703**, with an AUC of **0.77**, and loss curves confirmed stable convergence without divergence on validation data.

## Conclusion

This project provided hands-on experience with one-shot learning using Siamese networks. By systematically comparing optimizers, learning rates, and regularization, it demonstrated how architecture choices and hyperparameters interact to affect generalization, especially in low-data regimes. 
## Authors

- Roi Garber

- Nicole Kaplan
