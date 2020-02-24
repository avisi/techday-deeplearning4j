package nl.avisi.labs.deeplearning.transferlearning.handson.trainers;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.File;
import java.io.IOException;

public abstract class BaseTransferLearningTrainer {
    // sets the DL4J model-download folder to a local folder instead of default user home
    static {
        DL4JResources.setBaseDownloadURL("https://dl4jdata.blob.core.windows.net/");
        DL4JResources.setBaseDirectory(new File("dl4j-models/"));
    }

    protected long SEED = 12345; //Should normally be random, but is set in this case to be get reproducible results

    protected abstract void train() throws IOException;

    protected abstract String getModelname();

    protected abstract OutputLayer createOutputLayer();

    protected abstract ZooModel createModelFromZoo();

    /**
     * This method returns the distribution of weights to associate with the incoming connections
     * @return
     */
    protected NormalDistribution getDist() {
        return new NormalDistribution(0, 0.2 * (2.0 / (4096 + getNumberOfClasses())));
    }

    protected abstract int getNumberOfClasses();

    /***
     *
     * The fine tune configuration is used during the backpropagation phase to adjust weights.
     * The NesterovsUpdater is a popular approach.
     * @return
     */
    protected FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()//
                                                  .updater(new Nesterovs(5e-5))//
                                                  .seed(SEED).build();
    }

}
