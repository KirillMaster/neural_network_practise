namespace NeuralNetwork
{
    public interface ILossFunction
    {
        double LossFunction(double[][] batchOutputs, double[][] batchExpectedYs);
        double SampleLossFunction(double[] batchOutput, double[] expectedY);
        double LossFunctionDerivative(double output, double expectedY);
    }
}