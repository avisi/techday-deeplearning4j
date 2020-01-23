package nl.avisi.labs.deeplearning.transferlearning.model;
public class Prediction {

    private String label;
    private double percentage;

    public Prediction(String label, double percentage) {
        this.label = label;
        this.percentage = roundPercentage(percentage);
    }

    private double roundPercentage(final double percentage) {
        return Math.round(percentage * 100d) / 100d;
    }

    public String getLabel() {
        return label;
    }

    public double getPercentage() {
        return percentage;
    }

    public void setLabel(final String label) {
        this.label = label;
    }

    public void setPercentage(final double percentage) {
        this.percentage = percentage;
    }

    public String toString() {
        return String.format("%s: %.2f ", this.label, this.percentage);
    }

}
