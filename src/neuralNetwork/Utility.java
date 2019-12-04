package neuralNetwork;

public class Utility {

  public static double sigmoid(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  public static double dSigmoid(double x) {
    return x * (1 - x);
  }

  public static double randomBetween(double a, double b) {
    return a + (b - a) * Math.random();
  }
}
