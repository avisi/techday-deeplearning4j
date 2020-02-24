package nl.avisi.labs.deeplearning.transferlearning.handson.advanced;

import nl.avisi.labs.deeplearning.transferlearning.handson.trainers.BaseTransferLearningTrainer;

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.zoo.ZooModel;
import org.slf4j.Logger;

import java.io.IOException;

public class AdvancedTrainer extends BaseTransferLearningTrainer {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(AdvancedTrainer.class);

    private static final int NUM_CLASSES = 2; //Banana / No_banana
    private static final int TRAIN_PERC = 80; // Percentage of images that should be included in the trainings set, the rest is included in the test set
    private static final int BATCH_SIZE = 5;
    private static final int MAX_NUMBER_OF_ITERATIONS = 20; //20 should be a good amount to start with. Probably you need some more iterations to achieve a higher performance

    public static void main(String[] args) throws IOException {
        new AdvancedTrainer().train();
    }

    @Override
    protected void train() throws IOException {
        throw new NotImplementedException("Fix me");
    }

    @Override
    protected String getModelname() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    protected OutputLayer createOutputLayer() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    protected ZooModel createModelFromZoo() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    protected int getNumberOfClasses() {
        throw new NotImplementedException("Fix me");
    }

}
