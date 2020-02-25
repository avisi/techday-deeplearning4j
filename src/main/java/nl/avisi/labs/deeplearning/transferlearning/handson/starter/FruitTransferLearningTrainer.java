package nl.avisi.labs.deeplearning.transferlearning.handson.starter;

import nl.avisi.labs.deeplearning.transferlearning.handson.iterators.TransferLearningIterator;
import nl.avisi.labs.deeplearning.transferlearning.handson.trainers.BaseTransferLearningTrainer;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
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
    private static final int BATCH_SIZE = 5; // How many images are processed at a time
    private static final int MAX_NUMBER_OF_ITERATIONS = 40;

    public static void main(String[] args) throws IOException {
        new FruitTransferLearningTrainer().train();
    }


    @Override
    protected void train() throws IOException {
        // ZooModels are pre-defined networks; the model is downloaded from the specified location
        ZooModel zooModel = createModelFromZoo();


        // Initialize the pretrained network
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();


        // Deep learning has two phases:
        // 1. feature detection
        // 2. classification
        //
        // We will use transfer learning i.e. we re-purpose a network for our own use case. So instead of training a network from scratch we
        // re-use a trained network and adapt it by replacing the classification layer(s) with our own layers. So for instance in stead of training
        // a network to recognize 1000 categories we can remove the classification layer and add one which is used to recognize only 2 categories.
        // By 'freezing' anything below the classification layers we only need to train the classification layers.
        log.info(vgg16.summary());

        //create a new outputlayer for our own specific use case:

        Layer outputLayer = createOutputLayer();
        ComputationGraph vgg16Transfer = null;
        TransferLearningIterator iterator = null;

        if (outputLayer != null) {
            // the VGG16 network has 16 layers: 13 layers used for feature detection and 3 for classification. The last two layers are called "fc2" and "predictions" respectively.
            // We will freeze all layers up to and including "fc2" and replace the layer called "predictions". Frozen layers are not updated during backpropagation.
            vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)//
                                                                    // fine tuning is used during backpropagation
                                                    .fineTuneConfiguration(getFineTuneConfiguration())//
                                                    .setFeatureExtractor("fc2") //the specified layer and below are "frozen"
                                                    // the remaining layer is trained
                                                    .removeVertexKeepConnections("predictions") //replace the functionality of the final layer with out own layer
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
        // create an output layer. The  last-but-one layer of the VGG16 model has 4096 outgoing connections.
        // our new output layer has 2 outgoing signals (since we have two classes: banana or no_banana)
        // for classification purposes the SoftMax activation function is recommended as it results in a probability distribution
        // i.e. the sum of all scores in the output layer is 1
        // furthermore all values are in the interval [0,1> even for negative input values

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
