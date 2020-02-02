package tsutils;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

public class Utils {
	
	 public static BufferedImage drawDetectedObjects(BufferedImage img, List<DetectedObj> objList) {


	      BufferedImage result = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
	      Graphics g = result.getGraphics();
	      g.drawImage(img, 0, 0, null);

	      g.setColor(Color.yellow);

	      for(DetectedObj obj : objList){
	          Box box = obj.getBox();
	          int x = (int)(box.getLeft() * img.getWidth());
	          int y = (int)(box.getTop() * img.getHeight());
	          g.drawString(obj.getLabel(), x, y);
	          int width = (int)(box.getWidth() * img.getWidth());
	          int height = (int)(box.getHeight() * img.getHeight());
	          g.drawRect(x, y, width, height);
	      }

	      return result;
	  }
	 
	 
	  public static Tensor<UInt8> makeImageTensor(BufferedImage img) throws IOException {

	    	   /*     if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
	    	        	BufferedImage newimage =new BufferedImage(img.getWidth(),img.getHeight(), BufferedImage.TYPE_INT_RGB);
	    	        	newimage.getGraphics().drawImage(img,0,0,null);
	    	        	img = newimage;
	    	        }
	    	        
	    	        */
	     byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
	    	        // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
	     bgr2rgb(data);
	     final long BATCH_SIZE = 1;
	     final long CHANNELS = 3;
	     long[] shape = new long[] {BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
	     return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
	   }
	    	    
	
	   private static void bgr2rgb(byte[] data) {
	         for (int i = 0; i < data.length; i += 3) {
	           byte tmp = data[i];
	           data[i] = data[i + 2];
	            data[i + 2] = tmp;
	         }
	   }
	   
	   
	   public static  List<String>  getLabels(String dir) throws Exception, Exception {
		   extract("/models/labels.txt",dir);
		   return Files.readAllLines(new File(dir+"/labels.txt").toPath());

	   }
	   

	   public static List<DetectedObj> getDetections(List<Tensor<?>> outputs,List<String> labels) throws Exception {
		    List<DetectedObj> result = new ArrayList<>();
		    
		    try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
		            Tensor<Float> classesT = outputs.get(1).expect(Float.class);
		            Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
		           int maxObjects = (int) scoresT.shape()[1];
		           float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
		           float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
		           float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
		           for (int i = 0; i < scores.length; ++i) {
		               if (scores[i] < 0.5) {
		                   continue;
		               }
		               float score = scores[i];
		               float[] box = boxes[i];
		               
		               String label = labels.get((int) classes[i]);
		               float classeF = classes[i];
		               
		               DetectedObj detectedObj = new DetectedObj(label,score, box);
		               result.add(detectedObj);

		           }
		       }
			return result;
	   }
	   
	   
	   public static String extract(String jarFilePath,String temppath) throws Exception{

	        if(jarFilePath != null)
	        {
	            // Read the file we're looking for
	            InputStream fileStream = Utils.class.getResourceAsStream(jarFilePath);

	            // Was the resource found?
	            if(fileStream == null)
	                return null;

	            // Grab the file name
	            String[] chopped = jarFilePath.split("\\/");
	            String fileName = chopped[chopped.length-1];

	            // Create our temp file (first param is just random bits)
	            if (temppath == null) {
	            	Path path = Files.createTempDirectory("aats");
	            	temppath = path.toString();
	            }
	            File savedfile= new File(temppath+"/"+fileName);

	            // Create an output stream to barf to the temp file
	            OutputStream out = new FileOutputStream(savedfile);

	            // Write the file to the temp file
	            byte[] buffer = new byte[1024];
	            int len = fileStream.read(buffer);
	            while (len != -1) {
	                out.write(buffer, 0, len);
	                len = fileStream.read(buffer);
	            }


	            // Close the streams
	            fileStream.close();
	            out.close();
	            
	            
	            return temppath;
	   
	       }
			return jarFilePath;
	   }


}
