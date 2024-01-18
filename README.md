# AI Flower Classifier

This project was completed for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Image Classifier

The first part of this project creates and trains a neural network to correctly classify images of flowers.  This work is found in the [Image_Classifer_Project.ipynb](Image_Classifier_Project.ipynb) file and utilizes a Juypter notebook.  Transfer learning with alexnet is used to increase the accuracy of the neural network.

## Command Line Application

The second part of this project creates a command line application that trains a neural network and performs inference.  The file [train.py](train.py) includes the functions required to create, train, and save a neural network.  To increase the accuracy, you can select from two different Convolutional Neural Networks (CNNs) for transfer learning - alexnet and vgg13.  The file [predict.py](predict.py) includes the functions required to process an image as input, load a neural network, and perform inference to predict what flower is in the image.  