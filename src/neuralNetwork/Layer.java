package neuralNetwork;

import static neuralNetwork.Utility.*;

public class Layer {
  private double[] outputs, biases, deltas;
  private double[][] weights;
  private int noInputs, noOutputs;

  public Layer(int noInputs, int noOutputs) {
    this.noInputs = noInputs;
    this.noOutputs = noOutputs;
    outputs = new double[noOutputs];
    biases = new double[noOutputs];
    deltas = new double[noOutputs];
    weights = new double[noInputs][noOutputs];

    for (int i = 0; i < noOutputs; i++) {
      biases[i] = Math.random();
    }

    for (int i = 0; i < noInputs; i++) {
      for (int j = 0; j < noOutputs; j++) {
        weights[i][j] = randomBetween(-1, 1);
      }
    }
  }

  public void forwardPropagate(Layer previousLayer) {
    for (int j = 0; j < noOutputs; j++) {
      outputs[j] = 0.0;
      for (int i = 0; i < noInputs; i++) {
        outputs[j] += weights[i][j] * previousLayer.getOutput(i);
      }
      outputs[j] = sigmoid(outputs[j] + biases[j]);
    }
  }

  public void calculateDeltas(Layer nextLayer) {
    for (int i = 0; i < noOutputs; i++) {
      deltas[i] = 0;
      for (int j = 0; j < nextLayer.noOutputs; j++) {
        deltas[i] += nextLayer.getWeight(i, j) * nextLayer.getDelta(j);
      }
      deltas[i] *= dSigmoid(outputs[i]);
    }
  }

  public void update(Layer previousLayer, double learningRate) {
    for (int j = 0; j < noOutputs; j++) {
      biases[j] += learningRate * deltas[j];
    }

    for (int i = 0; i < noInputs; i++) {
      for (int j = 0; j < noOutputs; j++) {
        weights[i][j] += learningRate * previousLayer.getOutput(i) * deltas[j];
      }
    }
  }

  public double getOutput(int i) {
    return outputs[i];
  }

  public double getWeight(int i, int j) {
    return weights[i][j];
  }

  public double getDelta(int i) {
    return deltas[i];
  }

  public int getNoInputs() {
    return noInputs;
  }

  public int getNoOutputs() {
    return noOutputs;
  }

  public void setOutput(int i, double value) {
    outputs[i] = value;
  }

  public void setDelta(int i, double value) {
    deltas[i] = value;
  }
}
