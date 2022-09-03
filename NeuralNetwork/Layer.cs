using System;
using System.Linq;

namespace NeuralNetwork
{
    public class Layer
    {
        private Func<double, double> ActivationFunc { get; set; }
        
        private Func<double, double> ActivationFuncDerivative { get; set; }
        public int NeuronsCount { get; set; }
        private int PreviousLayerInputsCount { get; set; }
        private double[] PreviousLayerOutputs { get; set; }
        
        private double[] NextLayerDeltas { get; set; }
        
        private double Lambda { get; set; }
        
        private double[,] w { get; set; } 
        private double[] bias { get; set; } 

        public Layer(Func<double, double> activationFunc, Func<double, double> activationFuncDerivative, int neuronsCount, int previousLayerInputsCount, double lambda)
        {
            ActivationFunc = activationFunc;
            NeuronsCount = neuronsCount;
            PreviousLayerInputsCount = previousLayerInputsCount;
            Lambda = lambda;
            ActivationFuncDerivative = activationFuncDerivative;
            
            w = RandomHelper.FillRandomly(previousLayerInputsCount, neuronsCount);
            bias = RandomHelper.FillRandomly(neuronsCount);
        }

        private double[] Activation(double[] outputs)
        {
            return outputs.Select(ActivationFunc).ToArray();
        }

        public double[] Forward()
        {
            var result = new double[NeuronsCount];

            for (int i = 0; i < NeuronsCount; i++)
            {
                for (int j = 0; j < PreviousLayerInputsCount; j++)
                {
                    result[i] += PreviousLayerOutputs[j] * w[j,i] + bias[i];
                }
            }

            return Activation(result);
        }

        public void SetPreviousLayerOutputs(double[] previousLayerOutputs)
        {
            PreviousLayerOutputs = previousLayerOutputs;
        }

        public void SetNextLayerDeltas(double[] nextLayerDeltas)
        {
            NextLayerDeltas = nextLayerDeltas;
        }

        public double[] Backward()
        {
           ModifyWeights(NextLayerDeltas);
           return PreviousLayerDeltas(NextLayerDeltas);
        }
        
        private void ModifyWeights(double[] lastLayerDeltas)
        {
            for (int i = 0; i < PreviousLayerInputsCount; i++)
            {
                for (int j = 0; j < NeuronsCount; j++)
                {
                    w[i,j] = w[i,j] - Lambda * lastLayerDeltas[j] * PreviousLayerOutputs[i];
                    bias[j] = bias[j] - Lambda * lastLayerDeltas[j] * 1;
                } 
            }
        }
        
        private double[] PreviousLayerDeltas(double[] outputDeltas)
        {
            double[] previousLayerDeltas = new double[PreviousLayerInputsCount];
            for (int i = 0; i < PreviousLayerInputsCount; i++)
            {
                for (int j = 0; j < NeuronsCount; j++)
                {
                    previousLayerDeltas[i] += outputDeltas[j] * w[i, j] * ActivationFuncDerivative(PreviousLayerOutputs[i]);
                }
            }
            
            return previousLayerDeltas;
        }
    }
}