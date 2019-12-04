package neuralNetwork;

import mnist.MnistImageFile;
import mnist.MnistLabelFile;

import java.io.File;

public class SetCreator {

  public static SetPair createTrainSet(int noElements) {
    return createSet(noElements, "/res/trainImage.idx3-ubyte", "/res/trainLabel.idx1-ubyte");
  }

  public static SetPair createTestSet(int noElements) {
    return createSet(noElements, "/res/testImage.idx3-ubyte", "/res/testLabel.idx1-ubyte");
  }

  private static SetPair createSet(int noElements, String Image, String Label) {
    double[][] inputs = new double[noElements][28 * 28];
    double[][] outputs = new double[noElements][10];
    try {

      String path = new File("").getAbsolutePath();

      MnistImageFile m = new MnistImageFile(path + Image, "rw");
      MnistLabelFile l = new MnistLabelFile(path + Label, "rw");

      for (int i = 0; i < noElements; i++) {
        if (i % 100 == 0) {
          System.out.println("prepared: " + i);
        }

        outputs[i][l.readLabel()] = 1.0;
        for (int j = 0; j < 28 * 28; j++) {
          {
            inputs[i][j] = (double) m.read() / (double) 256;
          }
        }
        m.next();
        l.next();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
    return new SetPair(inputs, outputs);
  }
}
