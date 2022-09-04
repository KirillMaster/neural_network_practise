using System;

namespace NeuralNetwork.ActivationFunctions
{
    public class SoftMaxActivation : IActivationFunction
    {
        public double Activation(double val, double[] allLayer)
        {
            return ActivationSoftMax(val, allLayer);
        }

        public double ActivationDerivative(double val, double[] allLayer)
        {
            return ActivationSoftMaxDerivative(val, allLayer);
        }

        private static double ActivationSoftMax(double val, double[] allLayer)
        {
                
            double sum = 0; 
            for (int i = 0; i < allLayer.Length; i++)
            {
                sum += Math.Pow(Math.E, allLayer[i]);
            }
    
            return Math.Pow(Math.E, val) / sum;
        }

        private static double ActivationSoftMaxDerivative(double val, double[] allLayer)
        {
            var softMax = ActivationSoftMax(val, allLayer);
    
            return softMax * (1 - softMax); 
        }
    }
}