package nl.avisi.labs.deeplearning.transferlearning.handson.datahelpers;

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

public abstract class TransferLearningIterator {

    private static final Random rng = new Random(1663);

    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private InputSplit trainData, testData;

    private int batchSize;
    private int trainPercentage;

    /**
     * the
     * @param batchSize the number of files to read per batch
     * @param trainPercentage the percentage of data reserved for training purposes. The remainder will be used for testing
     */
    public TransferLearningIterator(int batchSize, int trainPercentage) {
        this.batchSize = batchSize;
        this.trainPercentage = trainPercentage;
    }

    private ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

    /**
     * The number of different classes the network must identify
     * @return the number of different classes the network must identify
     */
    protected abstract int getNumberOfClasses();

    /**
     * The labels which the network can assign to the outcome. The length of this list must
     * be in synch with the number of classes e.g. if the number of classes is 2 then the
     * list of labels must have two items.
     * The labels must be alphabetically sorted (i.e. match the order of subfolders)
     * @return a list of lables
     */
    public abstract List<String> getLabels();
    /**
     * The folder on the classpath where the dataset can be found.
     * The dataset must contain one subfolder for each class to distinguish
     * e.g. if number of classes is 2 then the dataset folder must have 2 subfolders.
     * Each subfolder will only be searched for images.
     * @return the number of different classes the network must identify
     */
    protected abstract String getDataSetFolder();


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
        //create the training data
        trainData = filesInDirSplit[0];
        //create the test data
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
