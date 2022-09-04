namespace NeuralNetwork.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Activation(double val, double[] foo);

        double ActivationDerivative(double val, double[] foo);
    }
}