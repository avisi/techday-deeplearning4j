package nl.avisi.labs.deeplearning.transferlearning.handson.advanced;

import nl.avisi.labs.deeplearning.transferlearning.handson.datahelpers.TransferLearningIterator;

import org.apache.commons.lang3.NotImplementedException;

import java.util.List;

public class AdvancedDataSetIterator extends TransferLearningIterator {

    public AdvancedDataSetIterator(int batchSize, int trainPercentage) {
        super(batchSize, trainPercentage);
        throw new NotImplementedException("Fix me");
    }


    @Override
    protected int getNumberOfClasses() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    //hier zou meer informatie bij kunnen
    protected String getDataSetFolder() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    public List<String> getLabels() {
        throw new NotImplementedException("Fix me");
    }

}
