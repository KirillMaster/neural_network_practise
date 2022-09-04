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
                error += expectedY[i] * Math.Log(outputNeurons[i]) +
                         (1 - expectedY[i]) * Math.Log(1 - outputNeurons[i]);
            }

            return  - (error / (double)outputNeurons.Length);
        }
        
        private static double CrossEntropyLossDerivative(double output, double expected)
        {
            return -(expected / output - (1 - expected) / (1 - output));
        }
    }
}