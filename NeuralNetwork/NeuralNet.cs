using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        private List<Layer> Layers { get; set; }
        private List<TrainData> TrainData { get; set; }
        private double EpochCount { get; set; }
        
        private double Accuracy { get; set; }
        private double Lambda { get; set; }
        
        private List<double> Errors { get; set; }

        public NeuralNet(List<TrainData> trainData, double lambda, double epochCount, double accuracy)
        {
            TrainData = trainData;
            Lambda = lambda;
            EpochCount = epochCount;
            Accuracy = accuracy;
        }

        public void Build()
        {
            Layers = new List<Layer>();

            var inputsCount = TrainData[0].X.Length;

            var layer1 = new Layer(ActivationFunc, ActivationDerivative, 6, inputsCount, Lambda);
            var layer2 = new Layer(ActivationFunc, ActivationDerivative, 1, layer1.NeuronsCount, Lambda);

            Layers.Add(layer1);
            Layers.Add(layer2);
        }

        private static double[] OutputDeltas(double[] outputNeurons, double[] expectedY)
        {
            double[] deltas = new double[outputNeurons.Length];
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                deltas[i] = 2 * (outputNeurons[i] - expectedY[i]) * ActivationDerivative(outputNeurons[i]);
            }

            return deltas;
        }


        public void Train()
        {
            double epochLoss = 10000000000;
            int k = 0;
           
            
            while (k < EpochCount && epochLoss > Accuracy)
            {
                k++;
                
                Errors = new List<double>();
                for (int i = 0; i < TrainData.Count; i++)
                {
                    var currentTrainPair = TrainData[i];
                    var output = Forward(currentTrainPair);
                    PrintCase(currentTrainPair, output);
                    Errors.Add(SampleError(output, currentTrainPair.ExpectedY));
                    Backward(output, currentTrainPair.ExpectedY);
       
                }

                epochLoss = EpochLoss();
                PrintEpochLoss(epochLoss);
                Console.WriteLine($"Epoch Count: {k + 1}");
            }
        }

        private void PrintCase(TrainData currentTrainPair, double[] output)
        {
            var input = string.Join(", ", currentTrainPair.X.Select(z => $"{z:f2}"));
            var result = string.Join(", ", output.Select(z => $"{z:f2}"));
            var expected = string.Join(", ", currentTrainPair.ExpectedY.Select(z => $"{z:f2}"));
            Console.WriteLine($"Input: {input}");
            Console.WriteLine($"Output: {result}");
            Console.WriteLine($"Expected: {expected}");
        }

        private void PrintEpochLoss(double loss)
        {
            Console.WriteLine($"EpochLoss: {loss}");
        }
        
        private static double SampleError(double[] outputNeurons, double[] expectedY)
        {
            double error = 0;
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                error += Math.Pow((outputNeurons[i] - expectedY[i]), 2);
            }

            return error;
        }


        private double EpochLoss()
        {
            double  sum = 0;
            for (int i = 0; i < Errors.Count; i++)
            {
                sum += Errors[i];
            }

            return sum;
        }


        private double[] Forward(TrainData currentTrainPair)
        {
            Layers[0].SetPreviousLayerOutputs(currentTrainPair.X);
            var output = Layers[0].Forward();

            for (int j = 1; j < Layers.Count; j++)
            {
                Layers[j].SetPreviousLayerOutputs(output);
                output = Layers[j].Forward();
            }

            return output;
        }

        private void Backward(double[] output, double[] expectedY)
        {
            var deltas = OutputDeltas(output, expectedY);
            var lastLayerIndex = Layers.Count - 1;


            for (int k = Layers.Count - 1; k >= 0; k--)
            {
                Layers[lastLayerIndex].SetNextLayerDeltas(deltas);
                deltas = Layers[lastLayerIndex].Backward();
            }
        }

        //RELU
        private static double ActivationFunc(double val)
        {
            //return val;
            return val >= 0 ? val : 0;
            //return (Math.Pow(Math.E, val) - Math.Pow(Math.E, -val)) / (Math.Pow(Math.E, val) + Math.Pow(Math.E, -val));

            //return 1 / (1 + Math.Pow(Math.E, -val));
        }

        //RELU'
        private static double ActivationDerivative(double val)
        {
            return 1;
            //return  1 - val * val;
            //return val * (1 - val);
        }
    }
}