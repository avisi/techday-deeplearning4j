package nl.avisi.labs.deeplearning.transferlearning.handson.advanced;

import nl.avisi.labs.deeplearning.transferlearning.handson.trainers.BaseTransferLearningTrainer;

import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.zoo.ZooModel;
import org.slf4j.Logger;

import java.io.IOException;

public class AdvancedTrainer extends BaseTransferLearningTrainer {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(AdvancedTrainer.class);

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
