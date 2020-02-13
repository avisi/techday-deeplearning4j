package nl.avisi.labs.deeplearning.transferlearning.dataHelpers;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;

public class FruitDataSetIterator extends TransferLearningIterator {
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng = new Random(1663);

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private InputSplit trainData, testData;

    public FruitDataSetIterator(int batchSize, int trainPercentage) {
        super(batchSize, trainPercentage);
    }

    public int getNumberOfClasses() {
        return 2;
    }

    public String getDataSetFolder() {
        return "/datasets/fruit/";
    }

    public List<String> getLabels() {
        return Arrays.asList("banana", "no_banana");
    }

}
