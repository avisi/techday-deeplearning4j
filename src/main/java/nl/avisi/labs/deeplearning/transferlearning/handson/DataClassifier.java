package nl.avisi.labs.deeplearning.transferlearning.handson;

import nl.avisi.labs.deeplearning.transferlearning.handson.model.Prediction;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public abstract class DataClassifier {

    protected abstract List<Prediction> classify(InputStream inputStream);

    protected ComputationGraph computationGraph;
    protected NativeImageLoader nativeImageLoader;
    protected List<String> labels;



    /**
     * Processes the image by feeding it through the network
     * @param image
     * @return
     */
    protected INDArray processImage(final INDArray image) {
        INDArray[] output = computationGraph.output(false, image);
        return output[0];
    }

    protected INDArray loadImage(final InputStream inputStream) {
        INDArray image = null;
        try {
            image = nativeImageLoader.asMatrix(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }

    protected abstract void normalizeImage(INDArray image);
}
