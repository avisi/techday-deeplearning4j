package nl.avisi.labs.deeplearning.transferlearning.dataHelpers;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Random;

public abstract class AbstractLearningIterator {

    private static final Random rng = new Random(1663);

    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private static final int NUM_CLASSES = 2;
    private InputSplit trainData, testData;

    private int batchSize;
    private int trainPercentage;

    AbstractLearningIterator(int batchSize, int trainPercentage) {
        this.batchSize = batchSize;
        this.trainPercentage = trainPercentage;
    }

    private ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

    abstract int getNumberOfClasses();

    abstract String getDataSetFolder();

    public abstract List<String> getLabels();

    public void setup() {
        File parentDir = null;
        try {
            parentDir = new File(this.getClass().getResource(getDataSetFolder()).toURI());
        } catch (URISyntaxException e) {
            e.printStackTrace();
        }

        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPercentage >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPercentage, 100 - trainPercentage);
        trainData = filesInDirSplit[0];
        testData = filesInDirSplit[1];
    }

    public DataSetIterator getTrainIterator() throws IOException {
        return makeIterator(trainData, batchSize);
    }

    public DataSetIterator getTestIterator() throws IOException {
        return makeIterator(testData, batchSize);

    }

    private DataSetIterator makeIterator(InputSplit split, int batchSize) throws IOException {
        DataSetIterator iter;
        try (ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker)) {
            recordReader.initialize(split);
            iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, getNumberOfClasses());
        }
        iter.setPreProcessor(new VGG16ImagePreProcessor());
        return iter;
    }

}
