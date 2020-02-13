package nl.avisi.labs.deeplearning.transferlearning.datahelpers;

        import java.util.Arrays;
        import java.util.List;

public class FlowerDataSetIterator extends TransferLearningIterator {

    public FlowerDataSetIterator(int batchSize, int trainPercentage) {
        super(batchSize, trainPercentage);
    }

    @Override
    int getNumberOfClasses() {
        return 5;
    }

    @Override
    String getDataSetFolder() {
        return "/datasets/flowers/";
    }

    public List<String> getLabels() {
        return Arrays.asList("daisy", "dandelion", "rose", "sunflower", "tulip");
    }

}
