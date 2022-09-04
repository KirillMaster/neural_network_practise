using System;
using System.Linq;
using NeuralNetwork.ActivationFunctions;

namespace NeuralNetwork
{
    public class Layer
    {
        private Func<double, double[], double> ActivationFunc { get; set; }
        
        public Func<double, double[],double> ActivationFuncDerivative { get; set; }
        private int NeuronsCount { get; set; }
        private int PreviousLayerOutputsCount { get; set; }
        private double[] PreviousLayerOutputs { get; set; }
        
        private double[] NextLayerDeltas { get; set; }
        
        private double Lambda { get; set; }
        
        private double[,] w { get; set; } 
        private double[] bias { get; set; } 

        public Layer(IActivationFunction activationFunc, int neuronsCount, double lambda)
        {
            ActivationFunc = activationFunc.Activation;
            NeuronsCount = neuronsCount;
            //PreviousLayerOutputsCount = previousLayerOutputsCount;
            Lambda = lambda;
            ActivationFuncDerivative = activationFunc.ActivationDerivative;
            
         
        }

        private double[] Activation(double[] outputs)
        {
            return outputs.Select(output => ActivationFunc(output, outputs)).ToArray();
        }

        public double[] Forward()
        {
            var result = new double[NeuronsCount];

            for (int i = 0; i < NeuronsCount; i++)
            {
                for (int j = 0; j < PreviousLayerOutputsCount; j++)
                {
                    result[i] += PreviousLayerOutputs[j] * w[j,i] + bias[i];
                }
            }

            return Activation(result);
        }

        public double[] Backward()
        {
           ModifyWeights(NextLayerDeltas);
           return Deltas(NextLayerDeltas);
        }
        
        private void ModifyWeights(double[] lastLayerDeltas)
        {
            for (int i = 0; i < PreviousLayerOutputsCount; i++)
            {
                for (int j = 0; j < NeuronsCount; j++)
                {
                    w[i,j] = w[i,j] - Lambda * lastLayerDeltas[j] * PreviousLayerOutputs[i];
                    bias[j] = bias[j] - Lambda * lastLayerDeltas[j] * 1;
                } 
            }
        }
        
        private double[] Deltas(double[] outputDeltas)
        {
            double[] previousLayerDeltas = new double[PreviousLayerOutputsCount];
            for (int i = 0; i < PreviousLayerOutputsCount; i++)
            {
                for (int j = 0; j < NeuronsCount; j++)
                {
                    previousLayerDeltas[i] += outputDeltas[j] * w[i, j] * ActivationFuncDerivative(PreviousLayerOutputs[i], PreviousLayerOutputs);
                }
            }
            
            return previousLayerDeltas;
        }
        
        public void SetPreviousLayerOutputs(double[] previousLayerOutputs)
        {
            PreviousLayerOutputs = previousLayerOutputs;
        }

        public void SetNextLayerDeltas(double[] nextLayerDeltas)
        {
            NextLayerDeltas = nextLayerDeltas;
        }

        public void SetPreviousLayerNeuronsCount(int previousLayerNeuronsCount)
        {
            PreviousLayerOutputsCount = previousLayerNeuronsCount;
        }

        public int GetNeuronsCount()
        {
            return NeuronsCount;
        }

        public void InitLayer()
        {
            w = RandomHelper.FillRandomly(PreviousLayerOutputsCount, NeuronsCount);
            bias = RandomHelper.FillRandomly(NeuronsCount);
        }
    }
}