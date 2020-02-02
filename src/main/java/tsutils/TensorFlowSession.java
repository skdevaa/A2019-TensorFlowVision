package tsutils;

import java.util.List;

import org.tensorflow.SavedModelBundle;

public class TensorFlowSession {
	
	private SavedModelBundle model;
	private String tempDir  ;
	private List<String> labels;
	
	
	public TensorFlowSession(SavedModelBundle model, String tempdir, List<String> labels) {
		this.model = model;
		this.tempDir = tempdir;
		this.labels = labels;
	}
	
	public String getTempDir() {
		return this.tempDir;
	}
	
	public SavedModelBundle getModel() {
		return this.model;
	}
	
	public List<String>  getLabels() {
		return this.labels;
	}

}
