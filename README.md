# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project: Build a Traffic Sign Recognition Program

### Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)


## Setup
### OS X and Linux
#### Install Anaconda
This project requires [Anaconda](https://www.continuum.io/downloads) and [Python 3.4](https://www.python.org/downloads/) or higher. If you don't meet all of these requirements, install the appropriate package(s).
#### Run the Anaconda Environment
Run these commands in your terminal to install all requirements:
```
$ git clone https://github.com/matthewzimmer/traffic-sign-classification.git
$ conda env create -f environment.yml
```

##### Install Tensorflow
###### GPU


```
$ source activate traffic-sign-classification
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.11.0-py3-none-any.whl
$ pip3 install --ignore-installed --upgrade $TF_BINARY_URL
```

NOTE:   This is still buggy. When I try loading tensorflow, it errors out saying it can't find libcudart.8.0.dylib.
        Problem is, I have CUDA 7.5 installed.

###### CPU
```
$ conda install --name traffic-sign-classification -c conda-forge tensorflow
```

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

```
$ conda install -c https://conda.anaconda.org/menpo opencv3
```

## Run the Notebook
### Start a Jupyter Server
Make sure to run the server from the same directory that you ran in the *Setup* steps above.

#### OS X and Linux
```
$ source activate traffic-sign-classification
$ jupyter notebook
```

## Useful Conda Commands

#### Update a conda environment

```
$ conda env update -f environment.yml
```


### Dataset

1. Download the dataset. You can download the pickled dataset in which we've already resized the images to 32x32 [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip).
 
2. Clone the project and start the notebook.
```
git clone https://github.com/udacity/traffic-signs
cd traffic-signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```
3. Follow the instructions in the `Traffic_Signs_Recognition.ipynb` notebook.

