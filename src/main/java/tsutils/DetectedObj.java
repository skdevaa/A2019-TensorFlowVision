package tsutils;



public class DetectedObj {
    private String label;
    private float score;

    private Box box = new Box();

    public DetectedObj(){

    }

    public DetectedObj(String label, float score, float[] box) {
        this.label = label;
        this.score = score;
        this.box = new Box(box);
    }
    
    public Box getBox() {
    	return this.box;
    }
    
    public float getScore() {
    	return this.score;
    }
    
    
    public String getLabel() {
    	return this.label;
    }
    
    public String toString() {
        return "{ label: " + label + ", score: " + score + ", box: " + box + " }";
    }


}