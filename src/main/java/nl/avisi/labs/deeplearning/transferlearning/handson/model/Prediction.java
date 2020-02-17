package nl.avisi.labs.deeplearning.transferlearning.handson.model;

public class Prediction {

    private String category;
    private double score;

    public Prediction(String category, double score) {
        this.category = category;
        this.score = score;
    }
    public String getCategory() {
        return category;
    }

    public void setCategory(final String category) {
        this.category = category;
    }

    public double getScore() {
        return score;
    }

    public void setScore(final double score) {
        this.score = score;
    }


}
