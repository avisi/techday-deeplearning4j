package nl.avisi.labs.deeplearning.transferlearning.handson;

import nl.avisi.labs.deeplearning.transferlearning.handson.model.Prediction;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

@Component
@Profile("tosti")
class TostiClassifier extends DataClassifier {
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;

    TostiClassifier() {
        try {
            File graphConfigurationFile = new File("TostiModel.zip");
            computationGraph = ModelSerializer.restoreComputationGraph("TostiModel.zip");
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
     * Normalize the image
     *
     * @param image
     */
    @Override
    protected void normalizeImage(final INDArray image) {
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
    }

}
