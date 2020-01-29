package nl.avisi.labs.deeplearning.transferlearning.trainers;

import nl.avisi.labs.deeplearning.transferlearning.dataHelpers.FacialExpressionsDataSetIterator;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

public class FacialExpressionsTransferLearningTrainer {
    // sets the DL4J model-download folder to al local folder instead of default user home
    static {
        DL4JResources.setBaseDownloadURL("https://dl4jdata.blob.core.windows.net/");
        DL4JResources.setBaseDirectory(new File("dl4j-models/"));
    }

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FacialExpressionsTransferLearningTrainer.class);

    private static final int NUM_CLASSES = 8;
    private static final long SEED = 12345; //Should normally be random, but is set in this case to get reproducible results

    private static final int TRAIN_PERC = 80; // Percentage of images that should be included in the trainings set, the rest is included in the test set
    private static final int BATCH_SIZE = 5;

    public static void main(String[] args) throws IOException {
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");

        //Import the VGG16 from the Model zoo
        ZooModel zooModel = VGG16.builder().build();
        //Initialize the pre-trained weights
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();

        log.info(vgg16.summary());
        OutputLayer outputLayer = new OutputLayer.Builder(NEGATIVELOGLIKELIHOOD)
                .nIn(4096)
                .nOut(NUM_CLASSES)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(getDist())
                .activation(Activation.SOFTMAX)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(getFineTuneConfiguration())
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions", outputLayer, "fc2").build();

        log.info(vgg16Transfer.summary());

        FacialExpressionsDataSetIterator.setup(BATCH_SIZE, TRAIN_PERC);
        DataSetIterator trainIter = FacialExpressionsDataSetIterator.trainIterator();
        DataSetIterator testIter = FacialExpressionsDataSetIterator.testIterator();

        Evaluation eval = vgg16Transfer.evaluate(testIter);
        log.info("Eval stats BEFORE fit....." + eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while (trainIter.hasNext()) {
            vgg16Transfer.fit(trainIter.next());

            if (iter % 10 == 0) {
                log.info("Evaluate model at iter " + iter + " ....");
                eval = vgg16Transfer.evaluate(testIter);
                log.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }
        File locationToSave = new File("FacialExpressionsModel.zip");
        ModelSerializer.writeModel(vgg16Transfer, locationToSave, false);
        ModelSerializer.addObjectToFile(locationToSave, "labels", FacialExpressionsDataSetIterator.getLabels());
    }

    @NotNull
    private static NormalDistribution getDist() {
        return new NormalDistribution(0, 0.2 * (2.0 / (4096 + NUM_CLASSES)));
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(5e-5))
                .seed(SEED).build();
    }
}
