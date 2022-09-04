using System;

namespace NeuralNetwork
{
    public static class ActivationFunctions
    {
            public static double ActivationRelu(double val, double[] foo)
            {
                return val;
            }
    
            public static double ActivationReluDerivative(double val, double[] foo)
            {
                return 1;
            }
            
            public static double ActivationSigmoid(double val, double[] foo)
            {
                return 1 / (1 + Math.Pow(Math.E, -val));
            }
    
            public static double ActivationSigmoidDerivative(double val, double[] foo)
            {
                return val * (1 - val);
            }
            
            public static double ActivationSoftMax(double val, double[] allLayer)
            {
                
                double sum = 0; 
                for (int i = 0; i < allLayer.Length; i++)
                {
                    sum += Math.Pow(Math.E, allLayer[i]);
                }
    
                return Math.Pow(Math.E, val) / sum;
            }
    
            public static double ActivationSoftMaxDerivative(double val, double[] allLayer)
            {
                var softMax = ActivationSoftMax(val, allLayer);
    
                return softMax * (1 - softMax); 
            }
    }
}