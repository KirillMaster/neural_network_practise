using System;

namespace NeuralNetwork.ActivationFunctions
{
    public class HyperbolicTanActivation : IActivationFunction
    {
        public double Activation(double val, double[] foo)
        {
            return (Math.Pow(Math.E, val) - Math.Pow(Math.E, -val)) / (Math.Pow(Math.E, val) + Math.Pow(Math.E, -val));
        }

        public double ActivationDerivative(double val, double[] foo)
        {
            return val * (1 - val);
        }
    }
}