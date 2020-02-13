package nl.avisi.labs.deeplearning.transferlearning;

import nl.avisi.labs.deeplearning.transferlearning.model.Prediction;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public interface DataClassifier {

    List<Prediction> classify(InputStream inputStream);
}
