

package com.automationanywhere.botcommand.sk;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.UInt8;

import tsutils.Box;
import tsutils.DetectedObj;

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
public class testtf {
  private static void printUsage(PrintStream s) {
    final String url =
        "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";
    s.println(
        "Java program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)");
    s.println("to label JPEG images.");
    s.println("TensorFlow version: " + TensorFlow.version());
    s.println();
    s.println("Usage: label_image <model dir> <image file>");
    s.println();
    s.println("Where:");
    s.println("<model dir> is a directory containing the unzipped contents of the inception model");
    s.println("            (from " + url + ")");
    s.println("<image file> is the path to a JPEG image file");
  }

  public static void main(String[] args) throws IOException {

    String modelDir = "C:\\Users\\Stefan Karsten\\Downloads\\newmodel7";
    String imageFile ="C:\\temp\\sample2.jpg";

  //  byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile));
   // Tensor<Float> image = constructAndExecuteGraphToNormalizeImageFloat(imageBytes);
    
    SavedModelBundle model = SavedModelBundle.load(modelDir,"serve");

    
    Image image = ImageIO.read(new File(imageFile));
    BufferedImage buffered = (BufferedImage) image;
    
  /*  final int width = buffered.getWidth();
    final int height = buffered.getHeight();
    BufferedImage newRGB = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    newRGB.createGraphics().drawImage(buffered, 0, 0, width, height, null);
    
    final int[] ib = ((DataBufferInt)((BufferedImage) newRGB).getRaster().getDataBuffer()).getData(); 
    
    	*/
    
   
    Tensor<UInt8> input = makeImageTensor(buffered);
        List<Tensor<?>> outputs = model
		        .session()
		        .runner()
		        .feed("image_tensor", input)
		        .fetch("detection_scores")
		        .fetch("detection_classes")
		        .fetch("detection_boxes")
		        .run();
    
    Tensor<Float> output0 =   (Tensor<Float>)outputs.get(0);
    
    System.out.println("Dim"+output0.numDimensions());
    System.out.println("Size"+output0.shape().length);
    
    List<String> labels = readAllLinesOrExit(Paths.get(modelDir, "labels.txt"));
    
    List<DetectedObj> result = new ArrayList<>();
    
    try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
            Tensor<Float> classesT = outputs.get(1).expect(Float.class);
            Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
           // All these tensors have:
           // - 1 as the first dimension
           // - maxObjects as the second dimension
           // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
           // This can be verified by looking at scoresT.shape() etc.
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
               
               System.out.println("Score "+score+" Class "+label);
               
               System.out.println("box"+box[0]+","+box[1]+","+box[2]+","+box[3]);

           }
       }

    BufferedImage imagenew = drawDetectedObjects(buffered, result);
    
    
    File outputfile = new File("C:\\\\temp\\\\persons_box.jpg");
    ImageIO.write(imagenew, "jpg", outputfile);
    
    
 /*   byte[] graphDef = readAllBytesOrExit(Paths.get(modelDir, "saved_model.pb"));
    List<String> labels =
        readAllLinesOrExit(Paths.get(modelDir, "labels.txt"));
    byte[] imageBytes = readAllBytesOrExit(Paths.get(imageFile));

    try (Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
      float[] labelProbabilities = executeInceptionGraph(graphDef, image);
      int bestLabelIdx = maxIndex(labelProbabilities);
      System.out.println(
          String.format("BEST MATCH: %s (%.2f%% likely)",
              labels.get(bestLabelIdx),
              labelProbabilities[bestLabelIdx] * 100f));
    }
    
    */
  }
  
  
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
    
    
  private static Tensor<Float> constructAndExecuteGraphToNormalizeImageFloat(byte[] imageBytes) {
    try (Graph g = new Graph()) {
      GraphBuilder b = new GraphBuilder(g);
      // Some constants specific to the pre-trained model at:
      // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
      //
      // - The model was trained with images scaled to 224x224 pixels.
      // - The colors, represented as R, G, B in 1-byte each were converted to
      //   float using (value - Mean)/Scale.
      final int H = 224;
      final int W = 224;
      final float mean = 117f;
      final float scale = 1f;

      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
      final Output<String> input = b.constant("input", imageBytes);
      final Output<Float> output =
          b.div(
              b.sub(
                  b.resizeBilinear(
                      b.expandDims(
                          b.cast(b.decodeJpeg(input, 3), Float.class),
                          b.constant("make_batch", 0)),
                      b.constant("size", new int[] {H, W})),
                  b.constant("mean", mean)),
              b.constant("scale", scale));
      try (Session s = new Session(g)) {
        // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
        return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
      }
    }
  }



  
  
  private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
	  
	  
    try (Graph g = new Graph()) {
      g.importGraphDef(graphDef);
      try (Session s = new Session(g);
          // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
          Tensor<Float> result =
              s.runner().feed("input", image).fetch("output").run().get(0).expect(Float.class)) {
        final long[] rshape = result.shape();
        if (result.numDimensions() != 2 || rshape[0] != 1) {
          throw new RuntimeException(
              String.format(
                  "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                  Arrays.toString(rshape)));
        }
        
        
        
        int nlabels = (int) rshape[1];
        return result.copyTo(new float[1][nlabels])[0];
      }
    }
  }

  private static int maxIndex(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }

  private static byte[] readAllBytesOrExit(Path path) {
    try {
      return Files.readAllBytes(path);
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(1);
    }
    return null;
  }

  private static List<String> readAllLinesOrExit(Path path) {
    try {
      return Files.readAllLines(path, Charset.forName("UTF-8"));
    } catch (IOException e) {
      System.err.println("Failed to read [" + path + "]: " + e.getMessage());
      System.exit(0);
    }
    return null;
  }

  // In the fullness of time, equivalents of the methods of this class should be auto-generated from
  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
  // like Python, C++ and Go.
  static class GraphBuilder {
    GraphBuilder(Graph g) {
      this.g = g;
    }

    Output<Float> div(Output<Float> x, Output<Float> y) {
      return binaryOp("Div", x, y);
    }

    <T> Output<T> sub(Output<T> x, Output<T> y) {
      return binaryOp("Sub", x, y);
    }

    <T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
      return binaryOp3("ResizeBilinear", images, size);
    }

    <T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
      return binaryOp3("ExpandDims", input, dim);
    }

    <T, U> Output<U> cast(Output<T> value, Class<U> type) {
      DataType dtype = DataType.fromClass(type);
      return g.opBuilder("Cast", "Cast")
          .addInput(value)
          .setAttr("DstT", dtype)
          .build()
          .<U>output(0);
    }

    Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
          .addInput(contents)
          .setAttr("channels", channels)
          .build()
          .<UInt8>output(0);
    }

    <T> Output<T> constant(String name, Object value, Class<T> type) {
      try (Tensor<T> t = Tensor.<T>create(value, type)) {
        return g.opBuilder("Const", name)
            .setAttr("dtype", DataType.fromClass(type))
            .setAttr("value", t)
            .build()
            .<T>output(0);
      }
    }
    Output<String> constant(String name, byte[] value) {
      return this.constant(name, value, String.class);
    }

    Output<Integer> constant(String name, int value) {
      return this.constant(name, value, Integer.class);
    }

    Output<Integer> constant(String name, int[] value) {
      return this.constant(name, value, Integer.class);
    }

    Output<Float> constant(String name, float value) {
      return this.constant(name, value, Float.class);
    }

    private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
    }

    private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
      return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
    }
    private Graph g;
  }
}

