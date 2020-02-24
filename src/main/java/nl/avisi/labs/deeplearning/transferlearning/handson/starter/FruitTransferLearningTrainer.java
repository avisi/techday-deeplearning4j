package nl.avisi.labs.deeplearning.transferlearning.handson.starter;

import nl.avisi.labs.deeplearning.transferlearning.handson.datahelpers.TransferLearningIterator;
import nl.avisi.labs.deeplearning.transferlearning.handson.trainers.BaseTransferLearningTrainer;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

import groovy.util.logging.Log4j;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;

@Log4j
public class FruitTransferLearningTrainer extends BaseTransferLearningTrainer {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(FruitTransferLearningTrainer.class);

    private static final int NUM_CLASSES = 2; //Banana / No_banana
    private static final int TRAIN_PERC = 80; // Percentage of images that should be included in the trainings set, the rest is included in the test set
    private static final int BATCH_SIZE = 5;
    private static final int MAX_NUMBER_OF_ITERATIONS = 40;

    public static void main(String[] args) throws IOException {
        new FruitTransferLearningTrainer().train();
    }


    @Override
    protected void train() throws IOException {
        // ZooModels are pre-defined networks; the model is downloaded from the specified location
        ZooModel zooModel = createModelFromZoo();

        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();

        log.info(vgg16.summary());
        //create a new outputlayer for our own specific use case:

        Layer outputLayer = createOutputLayer();
        ComputationGraph vgg16Transfer = null;
        TransferLearningIterator iterator = null;

        if (outputLayer != null) {
            vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)//
                                                    .fineTuneConfiguration(getFineTuneConfiguration())//
                                                    .setFeatureExtractor("fc2") //the specified layer and below are "frozen"
                                                    // the other layers are trained
                                                    .removeVertexKeepConnections("predictions") //replace the functionality of the final layer
                                                    .addLayer("predictions", outputLayer, "fc2").build();
            iterator = new FruitDataSetIterator(BATCH_SIZE, TRAIN_PERC);
            iterator.setup();
            DataSetIterator trainIter = iterator.getTrainIterator();
            DataSetIterator testIter = iterator.getTestIterator();

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
                if (!trainIter.hasNext() && iter < MAX_NUMBER_OF_ITERATIONS) {
                    trainIter.reset();
                } else if (iter == MAX_NUMBER_OF_ITERATIONS) {
                    break;
                }
            }
        }

        log.info(vgg16Transfer.summary());


        File locationToSave = new File(getModelname() + ".zip");
        ModelSerializer.writeModel(vgg16Transfer, locationToSave, false);
        if (outputLayer != null) {
            ModelSerializer.addObjectToFile(locationToSave, "labels", iterator.getLabels());
        }

    }

    @Override
    protected String getModelname() {
        return "FruitModel";
    }

    @Override
    protected OutputLayer createOutputLayer() {
        return new OutputLayer.Builder(NEGATIVELOGLIKELIHOOD)//
                                                             .nIn(4096)
                                                             .nOut(NUM_CLASSES)
                                                             .weightInit(getDist())
                                                             .activation(Activation.SOFTMAX)//
                                                             .build();
    }

    @Override
    protected ZooModel createModelFromZoo() {
        return VGG16.builder().build();
    }

    @Override
    protected int getNumberOfClasses() {
        return 2;
    }
}
