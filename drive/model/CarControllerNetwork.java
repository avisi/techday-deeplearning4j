package nl.avisi.labs.deeplearning.drive.model;

import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator.Set;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CarControllerNetwork {
	// sets the DL4J model-download folder to a local folder instead of default user home
	static { DL4JResources.setBaseDirectory(new File("dl4j-models/"));}
	private static final Logger log = org.slf4j.LoggerFactory.getLogger(CarControllerNetwork.class);

	// pseudo random number, if made random, each network created will be unique
	private static final int SEED = 123;
	private static final Set EMNIST_SET = Set.MNIST;
	private static final int NUM_CLASSES = EmnistDataSetIterator.numLabels(EMNIST_SET);
	private static final List<String> LABELS = EmnistDataSetIterator.getLabels(EMNIST_SET);

	// amount of images to test in each iteration
	private static final int BATCH_SIZE = 256;
	// height and width of the images
	private static final int HEIGHT = 28;
	private static final int WIDTH = 28;
	// amount of color channels, 1 = black/white, 3 would be RGB
	private static final int CHANNELS = 1;
	private static EmnistDataSetIterator emnistTrain, emnistTest;
	private MultiLayerNetwork network;

	private void setup() throws IOException {
		emnistTrain = new EmnistDataSetIterator(EMNIST_SET, BATCH_SIZE, true);
		emnistTest = new EmnistDataSetIterator(EMNIST_SET, BATCH_SIZE, false);
		initializeNetwork();
	}

	private void initializeNetwork() {
		// reduce the learning rate as the number of training epochs increases
        // iteration #, learning rate
        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        learningRateSchedule.put(0, 0.06);
        learningRateSchedule.put(50, 0.05);
        learningRateSchedule.put(200, 0.04);

		MultiLayerConfiguration networkConf = new NeuralNetConfiguration.Builder()
				.seed(SEED)
				.updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
				.weightInit(WeightInit.XAVIER)

				.list()
				.layer(new ConvolutionLayer.Builder(4, 4)
						.stride(1, 1)
						.activation(Activation.IDENTITY)
						.nOut(20)
						.build())				//TODO Assignment 1.2c
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(4,4)
						.stride(1, 1)
						.nOut(50)
						.activation(Activation.IDENTITY)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()
						.activation(Activation.RELU)
						.nOut(500)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX)
						.nOut(NUM_CLASSES)
						.build())
				.setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
				.build();

		network = new MultiLayerNetwork(networkConf);
        // TODO Assignment 1.2a
	            // TODO Assignment 1.2b
	            // TODO Assignment 1.2c
	            // TODO Assignment 1.2d
	            // TODO Assignment 1.2e
	}

	private void train() throws IOException {
		network.init();

		Evaluation evaluation;

		int iteration = 0;
		while(emnistTrain.hasNext()) {
			network.fit(emnistTrain.next());

			if(iteration % 10 == 0) {
				log.info("Evaluate model at iter "+iteration+" ....");
				evaluation = network.evaluate(emnistTest);
				log.info(evaluation.stats());
				emnistTest.reset();
			}

			iteration++;
		}

		log.info("Model build complete");
//Save the model
		File locationToSave = new File("DukeHandwritingNetwork.zip");
		boolean saveUpdater = true;
		ModelSerializer.writeModel(network, locationToSave, saveUpdater);
		ModelSerializer.addObjectToFile(locationToSave, "labels", LABELS);
		log.info("Model saved");
	}
	
	public static void main(String args[]) {
		try {
			CarControllerNetwork network = new CarControllerNetwork();
			network.setup();
			network.train();
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
}
