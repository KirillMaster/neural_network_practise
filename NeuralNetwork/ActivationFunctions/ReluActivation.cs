namespace NeuralNetwork.ActivationFunctions
{
    public class ReluActivation : IActivationFunction
    {
        public double Activation(double val, double[] foo)
        {
            return ActivationRelu(val, foo);
        }

        public double ActivationDerivative(double val, double[] foo)
        {
            return ActivationReluDerivative(val, foo);
        }
        
        public static double ActivationRelu(double val, double[] foo)
        {
            return val > 0 ? val : 0;
        }
    
        public static double ActivationReluDerivative(double val, double[] foo)
        {
            return val > 0 ? 1 : 0;
        }
    }
}