package weka.filters.unsupervised.attribute;

import weka.core.*;
import weka.filters.SimpleBatchFilter;
import weka.filters.SimpleStreamFilter;

import java.io.IOException;
import java.util.Enumeration;

/**
 * Created by mtdd1 on 16/11/17.
 */
public abstract class FirstFilterAbstract extends SimpleBatchFilter {


protected int st_index=1;

    @Override
    public String globalInfo() {
        return "trying My first filter";
    }

    @Override
    protected Instances determineOutputFormat(Instances instances) throws Exception {

        Instances result = new Instances(instances, 0);
        int numFeatures = getNumFeatures(instances);
        for (int index = numFeatures - 1; index >= 0; index--) {
            result.insertAttributeAt(new Attribute( "getFeatureName()"
                    + index), 1);
        }
        return result;

    }

    @Override
    protected Instances process(Instances instances) throws Exception {

        Instances result = new Instances (determineOutputFormat(instances),0);
        if(!isFirstBatchDone())		{
            this.initiliazeVectors(instances);
        }
        for(int i =0 ; i<instances.numInstances(); i++) {
            double[] features = getFeatures(instances.get(i));
            //double[] values = new double[result.numAttributes()];
            DenseInstance ins = new DenseInstance(0,features);
            //ins.setDataset(result);
            result.add(ins);
        }
        return result;
    }

    protected abstract void initiliazeVectors(Instances instances) throws Exception;

    protected  abstract  double[] getFeatures(Instance instance) throws IOException;
    protected  abstract  int getNumFeatures(Instances instance);
    @Override
    public Enumeration<Option> listOptions() {
        return Option.listOptionsForClass(this.getClass()).elements();
    }


    /* (non-Javadoc)
     * @see weka.filters.Filter#getOptions()
     */
    @Override
    public String[] getOptions() {
        return Option.getOptions(this, this.getClass());
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
        Option.setOptions(options, this, this.getClass());
    }


    /* (non-Javadoc)
     * @see weka.filters.Filter#getCapabilities()
     */
    @Override
    public Capabilities getCapabilities() {

        Capabilities result = new Capabilities(this);
        result.disableAll();

        // attributes
        result.enableAllAttributes();
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enableAllClasses();
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);

        result.setMinimumNumberInstances(0);

        return result;
    }
}
