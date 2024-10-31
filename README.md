# grocery-store-cnn 

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Overview

This project aims to develop a product classification system for grocery store shelves using convolutional neural networks. Such a system can be beneficial in practical applications, including:

	• Customer Assistance: Helping customers, including those with visual impairments, quickly locate specific products on shelves.

## 🎯 Project Objectives

	•	Single Product Classification: Implement a neural network model that recognizes individual products based on images taken from store shelves.
	•	Optimization through Fine-Tuning: Enhance the performance of pre-trained models like ResNet-18 to adapt them specifically for grocery products.

## 🖼️ Image Preprocessing

To improve model accuracy, images go through several preprocessing steps:

	1.	Resizing and Center Cropping: Images are resized to 224x224 pixels with a center crop to maintain aspect ratios.
	2.	Data Augmentation: Transformations such as RandomHorizontalFlip, RandomRotation, and ColorJitter are applied to make the model more robust to variations in lighting and orientation.

# Part 1: Initial Model Implementation

## 🧩 Model Architecture

The base model, GroceryModelFull, is inspired by the VGG architecture and includes:

	•	Three convolutional blocks with pooling layers.
	•	A global average pooling layer to reduce complexity.

This architecture balances representational capacity with computational efficiency, making it suitable for product images in a retail setting.

## 🔬 Ablation Study

An ablation study was conducted to understand the importance of various architectural components by creating the following model variations:

	•	GroceryModelNoBN: Base model without Batch Normalization.
	•	GroceryModelLessChannels: Model with reduced channels in each block.
	•	GroceryModelLessConvs: Model with a single convolution per block instead of multiple.
	•	GroceryModelLessBlocks: Model with two convolutional blocks instead of three.

The results of the ablation study provide insights into the impact of each architectural modification on model performance.

# Part 2: Fine-Tuning a Pretrained Network

In this part, we fine-tune a pretrained ResNet-18 model on the GroceryStoreDataset to improve classification accuracy for grocery products. This fine-tuning process is divided into two stages:

1.	Initial Fine-Tuning: Applying the training hyperparameters from the best model in Part 1.
2.	Hyperparameter Adjustment: Further tuning hyperparameters to achieve a validation accuracy target of 80%-90%.

## 🔧 Tuned Hyperparameters

To further enhance performance, specific adjustments were made to the model configuration:

1.	Fully Connected Layer with Dropout:
•	The fully connected layer (fc) was replaced with a Sequential block that includes Dropout layer (0.3) followed by a fully connected layer. The dropout helps reduce overfitting by randomly deactivating neurons during training.
2.	Batch Size Adjustment:
	•	The batch size was reduced from 64 to 32 to introduce more variability, helping the model avoid overfitting.

## 📊 Results

After applying the fine-tuning adjustments, the model showed a marked improvement in validation accuracy. The addition of a dropout layer and batch size adjustment helped the model generalize better to unseen data, achieving an accuracy within the target range on the validation set.
