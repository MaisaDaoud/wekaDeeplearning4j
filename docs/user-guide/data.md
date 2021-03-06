# Loading Data

The package provides so called `InstanceIterators` to load the given dataset in a correct shape. The following explains for which dataset and network type which iterator is necessary.

## DefaultInstanceIterator
The `DefaultInstanceIterator` assumes your dataset is of the following shape
```
@RELATION iris

@ATTRIBUTE sepallength  REAL
@ATTRIBUTE sepalwidth   REAL
@ATTRIBUTE petallength  REAL
@ATTRIBUTE petalwidth   REAL
@ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}

@DATA
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
...
```
that is, each row is represented as a vector without further interpretation
and you want to build a simple dense network of the form
``` 
DenseLayer -> DenseLayer -> ... -> OutputLayer
```

## ConvolutionInstanceIterator
To use convolutional neural networks in the case of a more sophisticated dataset, where the ARFF file represents column-wise flattened image pixels as e.g.:
```
@RELATION hotdog-3x3

@ATTRIBUTE pixel00  REAL
@ATTRIBUTE pixel01  REAL
@ATTRIBUTE pixel02  REAL
@ATTRIBUTE pixel10  REAL
@ATTRIBUTE pixel11  REAL
@ATTRIBUTE pixel12  REAL
@ATTRIBUTE pixel20  REAL
@ATTRIBUTE pixel21  REAL
@ATTRIBUTE pixel22  REAL

@ATTRIBUTE class    {hotdog, not-hotdog}

@DATA
127,32,15,234,214,144,94,43,23,hotdog
52,14,244,232,241,11,142,211,211,not-hotdog
...
```
it is necessary to set the iterator to `ConvolutionInstanceIterator`. With this, you have to set the `height`, `width` and `numChannels` so that the internals can interpret the flattened vector of each image in the ARFF file.

## ImageInstanceIterator
If the dataset consists of a set of image files it is necessary to prepare a meta-data ARFF file in the following format:
```
@RELATION mnist.meta.minimal

@ATTRIBUTE filename string
@ATTRIBUTE class {0,1,2,3,4,5,6,7,8,9}

@DATA
img_12829_0.jpg,0
img_32870_0.jpg,0
img_28642_0.jpg,0
...
```
This file informs the internals about the association between the image files and their labels. Additionally it is mandatory to set the iterator to `ImageInstanceIterator`. The setup of this iterator consist of 4 parameters:

- `height`: Height of the images
- `width`: Width of the images
- `numChannels`: Depth of the image (e.g.: RGB images have a depth of 3, whereas Grayscale images have a depth of 1)
- `imagesLocation`: The absolute path to the location of the images listed in the meta-data ARFF file