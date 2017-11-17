package weka.filters.unsupervised.attribute;

import org.junit.Test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
/**
 * JUnit tests for the Dl4jStringToWord2Vec Filter.
 *
 * @author Steven Lang
 */
public class FFTest {
    
    @Test
    public void testSDA() throws Exception {
        final String arffPath = "datasets/nominal/iris.arff";
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(arffPath);
        final Instances data = ds.getDataSet();
        FF sda = new FF();
        sda.setInputFormat(data);
        Instances d = Filter.useFilter(data,sda);
        System.out.println(d.toString());
    }

}
