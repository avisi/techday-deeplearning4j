package nl.avisi.labs.deeplearning.transferlearning.handson.model;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Prediction {

    private String label;
    private double percentage;

}
