package weka.classifiers.functions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.nd4j.linalg.activations.Activation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.dl4j.layers.*;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * JUnit tests applying the classifier to images converted to arff files.
 *
 * @author Steven Lang
 */
public class Dl4jMlpArffTest {

  /** Logger instance */
  private static final Logger logger = LoggerFactory.getLogger(Dl4jMlpArffTest.class);
  /** Current name */
  @Rule public TestName name = new TestName();
  /** Classifier */
  private Dl4jMlpClassifier clf;
  /** Start time for time measurement */
  private long startTime;

  @Before
  public void before() throws Exception {
    // Init mlp clf
    clf = new Dl4jMlpClassifier();
    clf.setSeed(TestUtil.SEED);
    clf.setNumEpochs(TestUtil.DEFAULT_NUM_EPOCHS);
    clf.setDebug(false);

    // Init data
    startTime = System.currentTimeMillis();
    //        TestUtil.enableUIServer(clf);
  }

  @After
  public void after() throws IOException {
    double time = (System.currentTimeMillis() - startTime) / 1000.0;
    logger.info("Testmethod: " + name.getMethodName());
    logger.info("Time: " + time + "s");
  }

  /**
   * Test minimal mnist dense net.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testMinimalMnistDenseArff() throws Exception {
    // Data
    Instances data = DatasetLoader.loadMiniMnistArff();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setNumChannels(1);
    cii.setHeight(28);
    cii.setWidth(28);
    cii.setTrainBatchSize(TestUtil.DEFAULT_BATCHSIZE);
    clf.setInstanceIterator(cii);

    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(256);
    denseLayer.setLayerName("Dense-layer");
    denseLayer.setActivationFn(Activation.RELU.getActivationFunction());

    DenseLayer denseLayer2 = new DenseLayer();
    denseLayer2.setNOut(128);
    denseLayer2.setLayerName("Dense-layer");
    denseLayer2.setActivationFn(Activation.RELU.getActivationFunction());

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
    outputLayer.setLayerName("Output-layer");

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

    clf.setNumEpochs(TestUtil.DEFAULT_NUM_EPOCHS);
    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(new Layer[] {denseLayer, denseLayer2, outputLayer});
    TestUtil.holdout(clf, data);
  }

  /**
   * Test minimal mnist conv net with arff file.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testMinimalMnistConvNetArff() throws Exception {

    Instances data = DatasetLoader.loadMiniMnistArff();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setNumChannels(1);
    cii.setHeight(28);
    cii.setWidth(28);
    cii.setTrainBatchSize(TestUtil.DEFAULT_BATCHSIZE);
    clf.setInstanceIterator(cii);

    int[] threeByThree = {3, 3};
    int[] twoByTwo = {2, 2};
    int[] oneByOne = {1, 1};
    List<Layer> layers = new ArrayList<>();

    ConvolutionLayer convLayer1 = new ConvolutionLayer();
    convLayer1.setKernelSize(threeByThree);
    convLayer1.setStride(oneByOne);
    convLayer1.setNOut(8);
    convLayer1.setLayerName("Conv-layer 1");
    layers.add(convLayer1);

    BatchNormalization bn1 = new BatchNormalization();
    bn1.setActivationFunction(Activation.RELU.getActivationFunction());
    layers.add(bn1);

    ConvolutionLayer convLayer2 = new ConvolutionLayer();
    convLayer2.setKernelSize(threeByThree);
    convLayer2.setStride(oneByOne);
    convLayer2.setActivationFn(Activation.RELU.getActivationFunction());
    convLayer2.setNOut(8);
    layers.add(convLayer2);

    BatchNormalization bn2 = new BatchNormalization();
    bn2.setActivationFunction(Activation.RELU.getActivationFunction());
    layers.add(bn2);

    SubsamplingLayer poolLayer1 = new SubsamplingLayer();
    poolLayer1.setPoolingType(PoolingType.MAX);
    poolLayer1.setKernelSize(twoByTwo);
    poolLayer1.setLayerName("Pool1");
    layers.add(poolLayer1);

    ConvolutionLayer convLayer3 = new ConvolutionLayer();
    convLayer3.setNOut(8);
    convLayer3.setKernelSize(threeByThree);
    layers.add(convLayer3);

    BatchNormalization bn3 = new BatchNormalization();
    bn3.setActivationFunction(Activation.RELU.getActivationFunction());
    layers.add(bn3);

    ConvolutionLayer convLayer4 = new ConvolutionLayer();
    convLayer4.setNOut(8);
    convLayer4.setKernelSize(threeByThree);
    layers.add(convLayer4);

    BatchNormalization bn4 = new BatchNormalization();
    bn4.setActivationFunction(Activation.RELU.getActivationFunction());
    layers.add(bn4);

    SubsamplingLayer poolLayer2 = new SubsamplingLayer();
    poolLayer2.setPoolingType(PoolingType.MAX);
    poolLayer2.setKernelSize(twoByTwo);
    layers.add(poolLayer2);

    BatchNormalization bn5 = new BatchNormalization();
    bn5.setActivationFunction(Activation.RELU.getActivationFunction());
    bn5.setDropOut(0.2);
    layers.add(bn5);

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
    outputLayer.setLossFn(new LossMCXENT());
    layers.add(outputLayer);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    nnc.setUseRegularization(true);

    clf.setNeuralNetConfiguration(nnc);
    Layer[] ls = new Layer[layers.size()];
    layers.toArray(ls);
    clf.setLayers(ls);

    TestUtil.holdout(clf, data);
  }
}
