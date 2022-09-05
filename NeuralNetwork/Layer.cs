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
        
        private double Gamma { get; set; }
        
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
                for (int j = 0; j < PreviousLayerOutputsCount; j++)
                {
                    result[i] += PreviousLayerOutputs[j] * w[j,i] + bias[i];
                }
                
                if (Double.IsNaN(result[i]))
                {
                    Console.WriteLine("Forward NAN catch");
                    throw new ApplicationException("Forward NAN");
                };
            }

       

            return Activation(result);
        }

        public double[] Backward()
        {
           GradientStep(NextLayerDeltas);
           return LocalGradient(NextLayerDeltas);
        }
        
        private void GradientStep(double[] lastLayerDeltas)
        {
            for (int i = 0; i < PreviousLayerOutputsCount; i++)
            {
                var activationFuncDerivative = ActivationFuncDerivative(PreviousLayerOutputs[i], PreviousLayerOutputs);
                for (int j = 0; j < NeuronsCount; j++)
                {
                    w[i,j] = w[i,j] - Lambda * lastLayerDeltas[j] * PreviousLayerOutputs[i] * activationFuncDerivative;
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
            double[] previousLayerDeltas = new double[PreviousLayerOutputsCount];
            for (int i = 0; i < PreviousLayerOutputsCount; i++)
            {
                for (int j = 0; j < NeuronsCount; j++)
                {
                    previousLayerDeltas[i] +=  outputDeltas[j] * w[i, j];
                }
            }
            
            return previousLayerDeltas;
        }
        
        public void SetPreviousLayerOutputs(double[] previousLayerOutputs)
        {
            PreviousLayerOutputs = previousLayerOutputs;
        }

        public void SetNextLayerLocalGradient(double[] nextLayerDeltas)
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
            bias = RandomHelper.FillRandomly(NeuronsCount, 1.0);
        }

        public void SetLambda(double lambda)
        {
            Lambda = lambda;
        }

        public void SetGamma(double gamma)
        {
            Gamma = gamma;
        }
    }
}