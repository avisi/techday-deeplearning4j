package nl.avisi.labs.deeplearning.transferlearning.trainers;

        import org.deeplearning4j.common.resources.DL4JResources;
        import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
        import org.deeplearning4j.nn.conf.layers.OutputLayer;
        import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
        import org.deeplearning4j.zoo.ZooModel;
        import org.jetbrains.annotations.NotNull;
        import org.nd4j.linalg.learning.config.Nesterovs;

        import java.io.File;
        import java.io.IOException;

public abstract class BaseTransferLearningTrainer {
    // sets the DL4J model-download folder to a local folder instead of default user home
    static {
        DL4JResources.setBaseDownloadURL("https://dl4jdata.blob.core.windows.net/");
        DL4JResources.setBaseDirectory(new File("dl4j-models/"));
    }

    protected static long SEED = 12345; //Should normally be random, but is set in this case to be get reproducible results


    protected abstract void train() throws IOException;

    protected abstract String getModelname();

    protected abstract OutputLayer createOutputLayer();

    protected abstract ZooModel createModelFromZoo();

    @NotNull
    protected NormalDistribution getDist() {
        return new NormalDistribution(0, 0.2 * (2.0 / (4096 + getNumberOfClasses())));
    }

    protected abstract int getNumberOfClasses();

    protected static FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()//
                                                  .updater(new Nesterovs(5e-5))//
                                                  .seed(SEED).build();
    }

}
