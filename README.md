# e4040-2019Fall-Project
e4040-2019Fall-Project is the seed repo for student projects - see README

This repository (in Github Classroom) is to serve as a seed for student projects githubs for e4040 2019 Fall.
To complete the project, the students need to create a copy of this repository and put their project work into that repository.
This github classroom assignment is a multi-person project - with the limit of three people. Be carefull in teaming up on the git.

INSTRUCTIONS for naming the students' solution repository for assignments with a team with several students, such as the final project. Students need to use a 4-letter groupID as a part of the name of the repo, as well as list all student UNIs: 
* Example: e4040-2019Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2019Fall-Project-MEME-zz9999-aa9999-aa0000.
* This change of the name can be done from the "Settings" tab which is located on the repo page.


ECBM4040 Group Project

Towards Accurate Binary Convolutional Neural Network

Team ZXCV: Qichen Hu qh2199, Xuechun Zhang xz2795, Yingtong Han yh3067


# Contents

There are six jupyter notebooks and one Python3 file within this folder (Keras version is another version of
code we used at first, which works but did not work well, so we just uploaded them here for your reference):

1. ResNet20_3_3.ipynb: implementation of ABC model based on ResNet20 (M=3, N=3)
2. ResNet20_5_3.ipynb: implementation of ABC model based on ResNet20 (M=5, N=3)
3. ResNet20_5_5.ipynb: implementation of ABC model based on ResNet20 (M=5, N=5)
4. ResNet34_3_3.ipynb: implementation of ABC model based on ResNet34 (M=3, N=3)
5. ResNet34_5_5.ipynb: implementation of ABC model based on ResNet34 (M=5, N=5)
6. ResNet34_M.ipynb: implementation of ABC model without binary activation based on ResNet20 (M=1, 3, 5)
6. utils_functions.py: functions used in building ResNet and ABC model, including creating tensorflow variables,
calling tensorflow built-in functions, and constructing ABC layer


# Description of key functions and features of each file:

1. Jupyter notebooks are files used for building model graphs and training. There is no additional function defined within those files.
2. The ResNet structures are built within each jupyter notebook. We first train full precision models (normal ResNet)
and use trained models’ weights as our initial values for ABC layers. We then train ABC-based ResNet from there.
3. utils_functions.py file contains functions used for creating tensorflow variables, calling tensorflow built-in functions,
and constructing ABC layer. For detailed usage of each function and explanations of the inputs and outputs, please refer to the comments.


# Instructions for running codes:

1. Set up the environment and install dependencies by running $ pip install -r requirement.txt in the command line.
2. Choose a model (ResNet20 or ResNet34) and a set of hyperparameters of M (number of binary filters) and N (number of binary activations).
Open corresponding jupyter notebook and click run all. Please note that you may only choose the two models mentioned above,
but you can change M and N to any positive integer values as you want. If you want to further play with more complex models,
you can tilt the ResNet structure within those jupyter notebooks.


# About the datasets

We use Cifar-10 as our training and testing dataset. Python will automatically download the dataset by running
“from tensorflow.keras.datasets import cifar10”. Please note that you can also test the file with MNIST dataset if you want.
The original ResNet structure performs better on MNIST but the performance of ABC-based ResNet is still relatively poor.

