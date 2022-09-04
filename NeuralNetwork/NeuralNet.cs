using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.ImageProcessing;
using NeuralNetwork.Sets;

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

        private int TestCount { get; set; }

        public NeuralNet(double lambda, double epochCount, double accuracy)
        {
            //TrainData = trainData;
            Lambda = lambda;
            EpochCount = epochCount;
            Accuracy = accuracy;
        }
        
        
        private double[] Forward(TrainData currentTrainPair)
        {
            var output = currentTrainPair.X;
            
            for (int j = 0; j < Layers.Count; j++)
            {
                Layers[j].SetPreviousLayerOutputs(output);
                output = Layers[j].Forward();
            }

            return output;
        }

        private void Backward(double[] output, double[] expectedY)
        {
            var deltas = OutputDeltas(output, expectedY);


            for (int k = Layers.Count-1; k >= 0; k--)
            {
                Layers[k].SetNextLayerDeltas(deltas);
                deltas = Layers[k].Backward();
            }
        }
        
        
        private double[] OutputDeltas(double[] outputNeurons, double[] expectedY)
        {
            double[] deltas = new double[outputNeurons.Length];
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                deltas[i] = LossFunctions.LossFuncDerivative(outputNeurons[i], expectedY[i]) * Layers[Layers.Count-1].ActivationFuncDerivative(outputNeurons[i], outputNeurons);
            }

            return deltas;
        }

        public void Build()
        {
            Layers = new List<Layer>();
            var inputsCount = TrainData[0].X.Length;
            var outputsCount = TrainData[0].ExpectedY.Length;
            
            TestCount = 2;
            Lambda = 0.1;
            Accuracy = 0.0001;
            EpochCount = 1000;
            
            var layer1 = new Layer(ActivationFunctions.ActivationSigmoid, ActivationFunctions.ActivationSigmoidDerivative, 3, inputsCount, Lambda);
            var layer2 = new Layer(ActivationFunctions.ActivationSigmoid, ActivationFunctions. ActivationSigmoidDerivative, outputsCount, layer1.NeuronsCount, Lambda);
            //var layer3 = new Layer(ActivationSoftMax, ActivationSoftMaxDerivative, outputsCount, layer2.NeuronsCount, Lambda);

            Layers.Add(layer1);
            Layers.Add(layer2);
            //Layers.Add(layer3);
       
        }

        public void CombineLayers()
        {
            
        }
        
        
        public void SetTrainData()
        {
            //TrainData = GetMultiplyTableTrain();
           //TrainData = GetXORTrain();
           TrainData = TestSets.GetTest();
           //TrainData = ImagesTrain();
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
                    // PrintCase(currentTrainPair, output);
                    Errors.Add(LossFunctions.LossFunction(output, currentTrainPair.ExpectedY));
                    Backward(output, currentTrainPair.ExpectedY);
       
                }

                epochLoss = EpochLoss();
                
                //if (k % 100 == 0)
               // {
                    PrintEpochLoss(epochLoss);
                //}
            }
            Console.WriteLine($"!=====================!");
            Console.WriteLine($"Final loss: {epochLoss}");
            Console.WriteLine($"Epoch Count: {k + 1}");
        }

        public void Test()
        {
            var random = new Random();
            
            for (int i = 0; i < TestCount; i++)
            {
                var num = random.Next(0, TrainData.Count);
                
                var test = TrainData[num];
                
                if (TrainData.Count < 10 &&  TrainData.Count >= TestCount)
                {
                    test = TrainData[i];
                }
                
                var output = Forward(test);
                Console.WriteLine("=======TEST========");
                
                PrintCase(test, output);
                Console.WriteLine($"Sample Error: {LossFunctions.LossFunction(output, test.ExpectedY)}");
            }
        }

        private void PrintCase(TrainData currentTrainPair, double[] output)
        {
            var input = string.Join(", ", currentTrainPair.X.Select(z => $"{z:f2}"));
            var result = string.Join(", ", output.Select(z => $"{z:f2}"));
            var expected = string.Join(", ", currentTrainPair.ExpectedY.Select(z => $"{z:f2}"));
            Console.WriteLine($"=====================");
           // Console.WriteLine($"Input: {input}");
            Console.WriteLine($"Output: {result}");
            Console.WriteLine();
            Console.WriteLine($"Expected: {expected}");
        }

        private void PrintEpochLoss(double loss)
        {
            Console.WriteLine($"EpochLoss: {loss}");
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
    }
}