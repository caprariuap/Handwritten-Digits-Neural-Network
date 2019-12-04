package neuralNetwork;

import java.util.Arrays;

public class Main {
  static int noTests = 5000, noTrainTests = 30000;

  public static void main(String[] args) {
    Network nn = new Network(0.03, 3, Arrays.asList(784, 30, 10));
    SetPair trainSet = SetCreator.createTrainSet(noTrainTests);
    nn.trainNetwork(trainSet.inputs, trainSet.outputs, 50);
    SetPair testSet = SetCreator.createTestSet(noTests);
    nn.testNetwork(testSet.inputs, testSet.outputs);
  }
}
