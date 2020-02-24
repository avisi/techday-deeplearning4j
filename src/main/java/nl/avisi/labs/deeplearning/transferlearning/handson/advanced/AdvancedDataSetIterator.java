package nl.avisi.labs.deeplearning.transferlearning.handson.advanced;

import nl.avisi.labs.deeplearning.transferlearning.handson.iterators.TransferLearningIterator;

import org.apache.commons.lang3.NotImplementedException;

import java.util.List;

public class AdvancedDataSetIterator extends TransferLearningIterator {

    public AdvancedDataSetIterator(int batchSize, int trainPercentage) {
        super(batchSize, trainPercentage);
    }

    @Override
    protected int getNumberOfClasses() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    protected String getDataSetFolder() {
        throw new NotImplementedException("Fix me");
    }

    @Override
    public List<String> getLabels() {
        //Hint: It is important to put the labels in alphabetical order
        throw new NotImplementedException("Fix me");
    }

}
