using System;

namespace NeuralNetwork.ActivationFunctions
{
    public class SigmoidActivation : IActivationFunction
    {
        public double Activation(double val, double[] foo)
        {
            return ActivationSigmoid(val, foo);
        }

        public double ActivationDerivative(double val, double[] foo)
        {
            return ActivationSigmoidDerivative(val, foo);
        }

        private static double ActivationSigmoid(double val, double[] foo)
        {
            return 1 / (1 + Math.Pow(Math.E, -val));
        }

        private static double ActivationSigmoidDerivative(double val, double[] foo)
        {
            return val * (1 - val);
        }
    }
}