using System;

namespace NeuralNetwork.LossFunctions
{
    public class MSELossFunction : LossFunctionBase, ILossFunction
    {
        public override double SampleLossFunction(double[] outputNeurons, double[] expectedY)
        {
            return MSE(outputNeurons, expectedY);
        }

        public double LossFunctionDerivative(double output, double expectedY)
        {
            return MSEDerivative(output, expectedY);
        }
        
        private static double MSE(double[] outputNeurons, double[] expectedY)
        {
            double error = 0;
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                error += Math.Pow((outputNeurons[i] - expectedY[i]), 2);
            }

            return  (error / (double)outputNeurons.Length);
        }
        
        private static double MSEDerivative(double output, double expected)
        {
            return 2 * (output- expected);
        }
    }
}