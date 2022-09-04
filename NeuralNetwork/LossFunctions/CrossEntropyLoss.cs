using System;

namespace NeuralNetwork.LossFunctions
{
    public class CrossEntropyLoss : ILossFunction
    {
        public double LossFunction(double[] outputNeurons, double[] expectedY)
        {
            return CrossEntropyLossFunction(outputNeurons, expectedY);
        }

        public double LossFunctionDerivative(double output, double expectedY)
        {
            return CrossEntropyLossDerivative(output, expectedY);
        }
        
        private static double CrossEntropyLossFunction(double[] outputNeurons, double[] expectedY)
        {
            double error = 0;
            
            
            for (int i = 0; i < outputNeurons.Length; i++)
            {

                var output = outputNeurons[i];
                if (output == 0)
                {
                    output = 0.0001;
                }
                
                error += expectedY[i] * Math.Log(output) +
                         (1 - expectedY[i]) * Math.Log(1 - output);
                
                var isNaN = Double.IsNaN(error);

                if (isNaN)
                {
                    Console.WriteLine($"NAN CATCH: Error: {error} ExpectedY: {expectedY[i]} Output: {outputNeurons[i]} ");
                    Console.WriteLine($"Log Math.Log(outputNeurons[i]): {Math.Log(outputNeurons[i])}");
                    Console.WriteLine($"Log Math.Log(1 - outputNeurons[i]) : {Math.Log(1 - outputNeurons[i])}");
                }
            }

            var result = -(error / (double) outputNeurons.Length);
           
            
            return result;
        }
        
        
        private static double CrossEntropyLossDerivative(double output, double expected)
        {
            if (Math.Abs(output - 1) < 1.0E-320 )
            {
                output = 0.99999999999;
            }

            if (Math.Abs(output) < 1.0E-320)
            {
                output = 0.000000000001;
            }
            
            return -(expected / output - (1 - expected) / (1 - output));
        }
    }
}