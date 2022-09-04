using System;

namespace NeuralNetwork
{
    public class LossFunctions
    {
        public static double LossFuncDerivative(double output, double expected)
        {
            return CrossEntropyLossDerivative(output, expected);
        }
        
        public static double LossFunction(double[] outputNeurons, double[] expectedY)
        {
            return CrossEntropyLossFunction(outputNeurons, expectedY);
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

        private static double MSELossFunction(double[] outputNeurons, double[] expectedY)
        {
            double error = 0;
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                error += Math.Pow((outputNeurons[i] - expectedY[i]), 2);
            }

            return  (error / (double)outputNeurons.Length);
        }

        private static double CrossEntropyLossDerivative(double output, double expected)
        {
            return -(expected / output - (1 - expected) / (1 - output));
        }

        private static double MSELossFuncDerivative(double output, double expected)
        {
            return 2 * (output- expected);
        }
    }
}