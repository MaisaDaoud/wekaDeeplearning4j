#!/bin/bash

java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier \
	 -S "42" \
	 -normalization "Standardize training data" \
	 -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -height 28 -width 28 -imagesLocation ../wekaDeeplearning4jCore/src/test/resources/nominal/mnist-minimal -numChannels 1 -bs 256 " \
	 -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 32 -kernelSizeX 3 -kernelSizeY 3 -strideX 1 -strideY 1" \
	 -layer "weka.dl4j.layers.SubsamplingLayer -kernelSizeX 2 -kernelSizeY 2 -strideX 2 -strideY 2" \
	 -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 64 -kernelSizeX 3 -kernelSizeY 3 -strideX 1 -strideY 1" \
	 -layer "weka.dl4j.layers.SubsamplingLayer -kernelSizeX 2 -kernelSizeY 2 -strideX 2 -strideY 2" \
	 -layer "weka.dl4j.layers.DenseLayer -nOut 512 -activation \"weka.dl4j.activations.ActivationReLU \" " \
	 -layer "weka.dl4j.layers.OutputLayer " \
     -numEpochs 10 \
     -t ../wekaDeeplearning4jCore/datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 66