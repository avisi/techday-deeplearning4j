package nl.avisi.labs.deeplearning.transferlearning.handson.starter;

        import nl.avisi.labs.deeplearning.transferlearning.handson.datahelpers.TransferLearningIterator;

        import java.util.Arrays;
        import java.util.List;

public class FlowerDataSetIterator extends TransferLearningIterator {

    public FlowerDataSetIterator(int batchSize, int trainPercentage) {
        super(batchSize, trainPercentage);
    }

    @Override
    protected int getNumberOfClasses() {
        return 5;
    }

    @Override
    protected String getDataSetFolder() {
        return "/datasets/flowers/";
    }

    public List<String> getLabels() {
        return Arrays.asList("daisy", "dandelion", "rose", "sunflower", "tulip");
    }

}
