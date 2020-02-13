package nl.avisi.labs.deeplearning.transferlearning;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.springframework.stereotype.Component;

import nl.avisi.labs.deeplearning.transferlearning.model.Prediction;

@Component
class FruitClassifier implements DataClassifier {
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private ComputationGraph computationGraph;
    private NativeImageLoader nativeImageLoader;
    private List<String> labels;

    FruitClassifier() {
        try {
            File graphConfigurationFile = new File("FruitModel.zip");
            computationGraph = ModelSerializer.restoreComputationGraph("FruitModel.zip");
            labels = ModelSerializer.getObjectFromFile(graphConfigurationFile, "labels");
        } catch (Exception e) {
            e.printStackTrace();
        }
        nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    }

    /**
     * Classify the image with the santa model
     *
     * @param inputStream
     * @return
     */
    public List<Prediction> classify(InputStream inputStream) {

        INDArray image = loadImage(inputStream);

        normalizeImage(image);

        INDArray output = processImage(image);

        List<Prediction> predictions = new ArrayList<Prediction>();
        for (int i = 0; i < labels.size(); i++) {
            predictions.add(new Prediction(labels.get(i), output.getFloat(i)));
        }

        return predictions;
    }

    /**
     * Processes the image by feeding it through the network
     * @param image
     * @return
     */
    private INDArray processImage(final INDArray image) {
        INDArray[] output = computationGraph.output(false, image);
        return output[0];
    }

    private INDArray loadImage(final InputStream inputStream) {
        INDArray image = null;
        try {
            image = nativeImageLoader.asMatrix(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    /**
     * Normalize the image
     *
     * @param image
     */
    private void normalizeImage(final INDArray image) {
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
    }

}
