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
        private int PreviousLayerNeuronsCount { get; set; }
        private double[] PreviousLayerNeurons { get; set; }
        private double[] NextLayerDeltas { get; set; }
        private double Lambda { get; set; }
        private double[,] w { get; set; } 
        private double[] bias { get; set; } 

        public Layer(IActivationFunction activationFunc, int neuronsCount)
        {
            ActivationFunc = activationFunc.Activation;
            NeuronsCount = neuronsCount; ;
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
                for (int j = 0; j < PreviousLayerNeuronsCount; j++)
                {
                    result[i] += PreviousLayerNeurons[j] * w[j,i] + bias[i];
                }
                ForwardIsNan(result[i]);
            }
            return Activation(result);
        }

        private void ForwardIsNan(double result)
        {
            if (Double.IsNaN(result))
            {
                Console.WriteLine("Forward NAN catch");
                throw new ApplicationException("Forward NAN");
            }
        }

        public double[] Backward()
        {
           GradientStep(NextLayerDeltas);
           return LocalGradient(NextLayerDeltas);
        }
        
        private void GradientStep(double[] lastLayerDeltas)
        {
            for (int i = 0; i < PreviousLayerNeuronsCount; i++)
            {
                var activationFuncDerivative = ActivationFuncDerivative(PreviousLayerNeurons[i], PreviousLayerNeurons);
                for (int j = 0; j < NeuronsCount; j++)
                {
                    w[i,j] = w[i,j] - Lambda * lastLayerDeltas[j] * PreviousLayerNeurons[i] * activationFuncDerivative;
                    bias[j] = bias[j] - Lambda * lastLayerDeltas[j] * 1 * activationFuncDerivative;
                    CheckNanAndThrow(w[i, j]);
                } 
            }
        }

        private void CheckNanAndThrow(double val)
        {
            if (Double.IsNaN(val))
            {
                throw new ApplicationException("Weight is NAN");
            }
        }
        
        private double[] LocalGradient(double[] outputDeltas)
        {
            double[] previousLayerDeltas = new double[PreviousLayerNeuronsCount];
            for (int i = 0; i < PreviousLayerNeuronsCount; i++)
            {
                for (int j = 0; j < NeuronsCount; j++)
                {
                    previousLayerDeltas[i] +=  outputDeltas[j] * w[i, j];
                }
            }
            
            return previousLayerDeltas;
        }
        
        public void SetPreviousLayerOutputs(double[] previousLayerNeurons)
        {
            PreviousLayerNeurons = previousLayerNeurons;
        }

        public void SetNextLayerLocalGradient(double[] nextLayerDeltas)
        {
            NextLayerDeltas = nextLayerDeltas;
        }

        public void SetPreviousLayerNeuronsCount(int previousLayerNeuronsCount)
        {
            PreviousLayerNeuronsCount = previousLayerNeuronsCount;
        }

        public int GetNeuronsCount()
        {
            return NeuronsCount;
        }

        public void InitLayer()
        {
            w = RandomHelper.FillRandomly(PreviousLayerNeuronsCount, NeuronsCount);
            bias = RandomHelper.FillRandomly(NeuronsCount, 1.0);
        }

        public void SetLambda(double lambda)
        {
            Lambda = lambda;
        }
    }
}