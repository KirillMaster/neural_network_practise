namespace NeuralNetwork
{
    public interface ILossFunction
    {
        public double LossFunction(double[] outputNeurons, double[] expectedY);
        public double LossFunctionDerivative(double output, double expectedY);
    }
}