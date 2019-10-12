# Human-Activity-Recognition-using-3D-CNN
# Human Activity Recognition using 3D-Convolutional Neural Network 
 Inplementation of 3D Convolutional Neural Network for video classification using [Keras](https://keras.io/)(with [tensorflow](https://www.tensorflow.org/) as backend).

## Description
This code requires [KTH dataset](http://www.nada.kth.se/cvap/actions/).
This code generates graphs of accuracy and loss, plot of model, result and class names as txt file and model as hd5.

This code is able to maximize a layer's output of any classification model.
(Only dense layer convolutional layer(2D/3D) and pooling layer(2D/3D) are allowed.)

## Requirements
python3  
opencv3, keras, numpy, Keras, Tensor Flow,   

## Options
Options of CNN.py are as following:  
`--batch`   batch size, default is 32  
`--epoch`   the number of epochs, 25, 50, 75, 100, 150  
`--videos`  a name of directory where dataset is stored   
`--nclass`  the number of classes you want to use, default is 06  
`--output`  a directory where the results described above will be saved  
`--color`   use RGB image or grayscale image, default is False  
`--depth`   the number of frames to use, default is 30  


You can see more information by using `--help` option
