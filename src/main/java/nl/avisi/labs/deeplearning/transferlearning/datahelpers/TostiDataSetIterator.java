package nl.avisi.labs.deeplearning.transferlearning.datahelpers;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class TostiDataSetIterator extends TransferLearningIterator {
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static final Random rng = new Random(1663);

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private InputSplit trainData, testData;

    public TostiDataSetIterator(int batchSize, int trainPercentage) {
        super(batchSize, trainPercentage);
    }

    /**
     * The network must decide whether an image is a banana or not, hence 2 classes
     * @return
     */
    public int getNumberOfClasses() {
        return 2;
    }

    public String getDataSetFolder() {
        return "/datasets/tostis/";
    }

    /**
     * An image contains either a banana or not
     * @return
     */
    public List<String> getLabels() {
        return Arrays.asList("no_tosti", "tosti");
    }

}
