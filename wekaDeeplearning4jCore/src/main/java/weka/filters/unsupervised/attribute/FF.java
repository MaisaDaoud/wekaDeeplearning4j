package weka.filters.unsupervised.attribute;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
//import org.deeplearning4j.models.*;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.DefaultInstanceIterator;
import weka.dl4j.text.sentenceiterator.WekaInstanceSentenceIterator;

import java.io.File;
import java.io.IOException;
import java.util.Enumeration;

/**
 * Created by mtdd1 on 16/11/17.
 */
public class FF extends  FirstFilterAbstract {
    int featureNum;
    int seed= 1;
    int iterations=1;
    private int listenerFreq = iterations/1;
    protected Instances m_trainData;
    protected  DataSetIterator m_trainIterator;
    protected AbstractInstanceIterator m_instanceIterator= new DefaultInstanceIterator();
    protected int m_queueSize = 10;
    protected  MultiLayerConfiguration conf;
    protected  MultiLayerNetwork model;
    public FF(){
        super();
    }
    @Override
    protected double[] getFeatures(Instance instance) throws IOException {

//        double []temp;
       double [] result= new double[this.featureNum];
//
//
//            int rand1 = (int)(Math.random()*4);
//            int rand2 = (int)(Math.random()*4);
//            int rand3 = (int)(Math.random()*4);
//            temp = instance.get(0).toDoubleArray();
//            int l= temp.length;
//            result[0]=temp[rand1];
//            result[1]=temp[rand2];
//            result[2]=temp[rand3];

        //Load the model
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork("./datasets/Models/MyMultiLayerNetwork.zip");
        INDArray ndarr= Nd4j.create(instance.toDoubleArray());
        long[] offsets = {0};
        int[] shape = {1};
        int[] stride={3};
          //NDArrayIndex.interval(0,3);
        INDArray temp=restored.getLayer(0).preOutput(ndarr.getRow(0).get(NDArrayIndex.interval(0,4)));//ndarr.getRow(0).getColumn(4).de);
        for(int i=1; i<(restored.getnLayers()/2);i++){
            temp=restored.getLayer(i).preOutput(temp);


        }
        for(int i =0 ;i<temp.length();i++){
            result[i]=temp.getDouble(i);
        }

        return result;
    }

    @Override
    protected void initiliazeVectors(Instances instances) throws Exception{

       // SentenceIterator m_trainIterator = new WekaInstanceSentenceIterator(instances,this.st_index-1);
       m_trainIterator = getDatasetIerator(instances);

        //log.info("Build model....");
        this.conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(instances.numAttributes()-1).nOut(instances.numAttributes()*100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(1, new RBM.Builder().nIn(instances.numAttributes()*100).nOut(instances.numAttributes()*50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(2, new RBM.Builder().nIn(instances.numAttributes()*50).nOut(instances.numAttributes()*25).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(3, new RBM.Builder().nIn(instances.numAttributes()*25).nOut(instances.numAttributes()*10).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(4, new RBM.Builder().nIn(instances.numAttributes()*10).nOut((int)(instances.numAttributes()-1*.5)).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //encoding stops
                .layer(5, new RBM.Builder().nIn((int)(instances.numAttributes()-1*.5)).nOut(instances.numAttributes()*10).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build()) //decoding starts
                .layer(6, new RBM.Builder().nIn(instances.numAttributes()*10).nOut(instances.numAttributes()*25).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(7, new RBM.Builder().nIn(instances.numAttributes()*25).nOut(instances.numAttributes()*50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(8, new RBM.Builder().nIn(instances.numAttributes()*50).nOut(instances.numAttributes()*100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(instances.numAttributes()*100).nOut(instances.numAttributes()-1).build())
                .pretrain(true).backprop(true)
                .build();

        this.model = new MultiLayerNetwork(this.conf);
        this.model.init();

        this.model.setListeners(new ScoreIterationListener(listenerFreq));
       // this.model.fit();
        //log.info("Train model....");
       while(m_trainIterator.hasNext()) {
            DataSet next = m_trainIterator.next();
            model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
        }
        //Save the model
        File locationToSave = new File("./datasets/Models/MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        //log.info("Train model....");
        /*while(iter.hasNext()) {
            DataSet next = iter.next();
            model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
        }*/
        // sets the tokenizer
       // this.m_tokenizerFactory.setTokenPreProcessor(this.m_preprocessor);

        // initializes stopwords
       // this.m_stopWordsHandler.initialize();

        // Building model
//        this.vec = new Word2Vec.Builder()
//                .minWordFrequency(this.m_minWordFrequency)
//                .useAdaGrad(this.m_useAdaGrad)
//                .allowParallelTokenization(this.m_allowParallelTokenization)
//                .enableScavenger(this.m_enableScavenger)
//                .negativeSample(this.m_negative)
//                .sampling(this.m_sampling)
//                .epochs(this.m_epochs)
//                .learningRate(this.m_learningRate)
//                .minLearningRate(this.m_minLearningRate)
//                .workers(this.m_workers)
//                .iterations(this.m_iterations)
//                .layerSize(this.m_layerSize)
//                .seed(this.m_seed)
//                .windowSize(this.m_windowSize)
//                .iterate(iter)
//                .stopWords(this.m_stopWordsHandler.getStopList())
//                .tokenizerFactory(this.m_tokenizerFactory)
//                .build();
//
//        // fit model
//        this.vec.fit();
    }

    @Override
    protected int getNumFeatures(Instances instances) {
        this.featureNum=(int)(instances.numAttributes()-1*.5);
        return featureNum;
    }
    /* (non-Javadoc)
	 * @see weka.filters.Filter#listOptions()
	 */
    @Override
    public Enumeration<Option> listOptions() {
        //this.getClass().getSuperclass()
        return Option.listOptionsForClassHierarchy(this.getClass(), this.getClass().getSuperclass()).elements();
    }


    /* (non-Javadoc)
     * @see weka.filters.Filter#getOptions()
     */
    @Override
    public String[] getOptions() {
        return Option.getOptionsForHierarchy(this, this.getClass().getSuperclass());

        //return Option.getOptions(this, this.getClass());
    }


    /**
     * Parses the options for this object.
     *
     *
     * @param options
     *            the options to use
     * @throws Exception
     *             if setting of options fails
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        Option.setOptionsForHierarchy(options, this, this.getClass().getSuperclass());
        // Option.setOptions(options, this, this.getClass());
    }
    public DataSetIterator getDatasetIerator(Instances instances) throws  Exception{
        instances.setClassIndex(instances.numAttributes()-1); //supposed that the class index is the last attr.
        DataSetIterator iter = m_instanceIterator.getDataSetIterator(instances, this.seed);
        if (m_queueSize > 0) {
            iter = new AsyncDataSetIterator(iter, m_queueSize);
        }
        return iter;
    }
    @OptionMetadata(description = "The queue size for asynchronous data transfer (default: 0, synchronous transfer).",
            displayName = "queue size for asynchronous data transfer", commandLineParamName = "queueSize",
            commandLineParamSynopsis = "-queueSize <int>", displayOrder = 9)
    public void setQueueSize(int QueueSize) {
        m_queueSize = QueueSize;
    }

    public static void manin(String[] args){runFilter(new FF(),args);}
}
