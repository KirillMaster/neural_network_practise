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
            if (val < 1.0E-20)
            {
                val = 0.01;
            }
            
            for (int i = 0; i < allLayer.Length; i++)
            {

                if (allLayer[i] < 1.0E-20)
                {
                    allLayer[i] = 0.01;
                }
                
                var exp = Math.Pow(Math.E, allLayer[i]);
                sum += exp;
                
                // var isNaN = Double.IsNaN(sum);
                //
                // if (isNaN)
                // {
                //     Console.WriteLine($"NAN CATCH SoftMax: : {sum} Caused by: {allLayer[i]}");
                // }
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