package neuralNetwork;

import java.util.ArrayList;
import java.util.List;

import static neuralNetwork.Utility.*;

public class Network {
  private final List<Layer> layers = new ArrayList<>();
  private double learningRate;

  public Network(double learningRate, int noLayers, List<Integer> layerSizes) {
    this.learningRate = learningRate;
    int prevSize = 0;
    for (int i = 0; i < noLayers; i++) {
      layers.add(new Layer(prevSize, layerSizes.get(i)));
      prevSize = layerSizes.get(i);
    }
  }

  private void predictOutputs(double[] inputs) {
    Layer inputLayer = getInputLayer();
    int outputIndex = inputLayer.getNoOutputs();
    for (int i = 0; i < outputIndex; i++) {
      inputLayer.setOutput(i, inputs[i]);
    }

    for (int i = 1; i < layers.size(); i++) {
      Layer currentLayer = layers.get(i);
      Layer previousLayer = layers.get(i - 1);
      currentLayer.forwardPropagate(previousLayer);
    }
  }

  public void trainNetwork(double[][] inputs, double[][] targets, int maximum) {
    for (int t = 0; t < maximum; t++) {
      System.out.println("trained: " + t);
      for (int i = 0; i < inputs.length; i++) {
        predictOutputs(inputs[i]);
        Layer outputLayer = getOutputLayer();
        for (int j = 0; j < outputLayer.getNoOutputs(); j++) {
          outputLayer.setDelta(
              j, dSigmoid(outputLayer.getOutput(j) * (targets[i][j] - outputLayer.getOutput(j))));
        }
        outputLayer.update(layers.get(layers.size() - 2), learningRate);
        for (int j = layers.size() - 2; j > 0; j--) {
          Layer currentLayer = layers.get(j);
          Layer previousLayer, nextLayer;
          previousLayer = layers.get(j - 1);
          nextLayer = layers.get(j + 1);
          currentLayer.calculateDeltas(nextLayer);
          currentLayer.update(previousLayer, learningRate);
        }
      }
    }
  }

  public int showResult(double[] input) {
    predictOutputs(input);
    Layer outputLayer = getOutputLayer();
    if (outputLayer.getNoOutputs() == 1) {
      return outputLayer.getOutput(0) < 0.5 ? 0 : 1;
    }
    int pos = 0;
    double max = outputLayer.getOutput(0);
    for (int i = 1; i < outputLayer.getNoOutputs(); i++)
      if (outputLayer.getOutput(i) > max) {
        max = outputLayer.getOutput(i);
        pos = i;
      }
    return pos;
  }

  public void testNetwork(double[][] inputs, double[][] outputs) {
    int correct, incorrect;
    correct = incorrect = 0;
    for (int i = 0; i < inputs.length; i++) {
      int poz = 0;
      double max = outputs[i][0];
      for (int j = 1; j < outputs[i].length; j++)
        if (outputs[i][j] > max) {
          max = outputs[i][j];
          poz = j;
        }
      if (showResult(inputs[i]) == poz) correct++;
      else {
        incorrect++;
      }
    }
    System.out.println((double) 100 * correct / (correct + incorrect));
  }

  Layer getInputLayer() {
    return layers.get(0);
  }

  Layer getOutputLayer() {
    return layers.get(layers.size() - 1);
  }
}
