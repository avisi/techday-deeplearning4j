package nl.avisi.labs.deeplearning.transferlearning.datahelpers;

import org.apache.commons.lang3.NotImplementedException;

import java.util.List;

public class DefaultDataSetIterator extends TransferLearningIterator {

    public DefaultDataSetIterator(int batchSize, int trainPercentage) {
        super(batchSize, trainPercentage);
        throw new NotImplementedException("Fix me");
    }


    @Override
    int getNumberOfClasses() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    String getDataSetFolder() {
        throw new NotImplementedException("Fix me");
    }

    public List<String> getLabels() {
        throw new NotImplementedException("Fix me");
    }

}
