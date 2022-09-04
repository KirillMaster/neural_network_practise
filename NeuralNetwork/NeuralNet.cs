using System;
using System.Collections.Generic;
using System.Linq;
using Accord;
using Accord.Math;
using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.ImageProcessing;
using NeuralNetwork.LossFunctions;
using NeuralNetwork.Sets;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        private List<Layer> Layers { get; set; }
        private List<Batch> TrainBatches { get; set; }
        private double EpochCount { get; set; }
        
        private double Accuracy { get; set; }
        private double Lambda { get; set; }
        
        private List<double> Errors { get; set; }

        private int TestCount { get; set; }
        
        private ILossFunction LossFunction { get; set; }

        private double Gamma { get; set; }

        private int BatchSize { get; set; } = 1;

        public NeuralNet(double lambda, double epochCount, double accuracy)
        {
            Lambda = lambda;
            EpochCount = epochCount;
            Accuracy = accuracy;
        }


        private double[][] Forward(Batch currentBatch)
        {
            var batchOutputs = new double[currentBatch.Size][];
            
            for (int i = 0; i < currentBatch.Size; i++)
            {
                batchOutputs[i] = ForwardForOneSample(currentBatch.TrainData[i]);
            }

            return batchOutputs;
        }
        
        
        private double[] ForwardForOneSample(TrainData currentTrainPair)
        {
            var output = currentTrainPair.NormalizedX;
            
            for (int j = 0; j < Layers.Count; j++)
            {
                Layers[j].SetPreviousLayerOutputs(output);
                output = Layers[j].Forward();
            }

            return output;
        }

        private void Backward(double[][] batchOutputs, double[][] batchExpectedYs)
        {
            var deltas = OutputDeltas(batchOutputs, batchExpectedYs);


            for (int k = Layers.Count-1; k >= 0; k--)
            {
                Layers[k].SetNextLayerDeltas(deltas);
                deltas = Layers[k].Backward();
            }
        }
        
        private double[] OutputDeltas(double[][] outputNeurons, double[][] expectedY)
        {
            double[] deltas = new double[BatchSize];
            for (int i = 0; i < BatchSize; i++)
            {
                for (int j = 0; j < outputNeurons[0].Length; j++)
                {
                    deltas[i] += 
                        LossFunction.LossFunctionDerivative(outputNeurons[i][j], expectedY[i][j]) 
                        * Layers[^1].ActivationFuncDerivative(outputNeurons[i][j], outputNeurons[i]);
                }
            }

            return deltas;
        }

        public void Build()
        {
            var outputsCount = TrainBatches[0].ExpectedYs[0].Length;
            Accuracy = 0.0001;
            TestCount = 2;
            
            Lambda = 0.1;
            
            Gamma = 0.8;
            EpochCount = 100000;
  
            
            LossFunction = new MSELossFunction();

            var layers = new List<Layer>()
            {
               // new Layer(new HyperbolicTanActivation(), 7),
                new Layer(new SigmoidActivation(), 3),
                new Layer(new SigmoidActivation(), outputsCount)
            };

            CombineLayers(layers);
        }

        private void CombineLayers(List<Layer> layers)
        {
            if (layers.Count == 0)
            {
                return;
            }
            
            var inputsCount = TrainBatches[0].TrainData[0].X.Length;
            var previousLayerOutputsCount = inputsCount;
            
            foreach (var layer in layers)
            {
                layer.SetPreviousLayerNeuronsCount(previousLayerOutputsCount);
                layer.SetLambda(Lambda);
                layer.SetGamma(Gamma);
                layer.InitLayer();
                previousLayerOutputsCount = layer.GetNeuronsCount();
            }

            Layers = layers;
        }
        
        
        public void SetTrainData()
        {
            //TrainData = TestSets.GetMultiplyTableTrain();
           //TrainData = TestSets.GetXORTrain();
           TrainBatches = TestSets.GetTest(BatchSize);
           //TrainBatches = TestSets.ImagesTrain(BatchSize);
        }


        public void Train()
        {
            double epochLoss = 10000000000;
            int k = 0;
           
            
            while (k < EpochCount && epochLoss > Accuracy)
            {
                k++;

                TrainBatches.Shuffle();
                
                Errors = new List<double>();
                for (int i = 0; i < TrainBatches.Count; i++)
                {
                    var currentTrainBatch = TrainBatches[i];
                    var output = Forward(currentTrainBatch);
                    // PrintCase(currentTrainPair, output);
                    Errors.Add(LossFunction.LossFunction(output, currentTrainBatch.ExpectedYs));
                    Backward(output, currentTrainBatch.ExpectedYs);
       
                }

                epochLoss = EpochLoss();

                if (k % 10 == 0)
                {
                    //breakpoint
                    PrintEpochLoss(epochLoss);
                }
                
                //if (k % 100 == 0)
               // {
                   // PrintEpochLoss(epochLoss);
                //}
            }
            Console.WriteLine($"!=====================!");
            Console.WriteLine($"Final loss: {epochLoss}");
            Console.WriteLine($"Epoch Count: {k}");
        }

        public void Test()
        {
            var random = new Random();
            
            for (int i = 0; i < TestCount; i++)
            {
                var trainData = TrainBatches.SelectMany(x => x.TrainData).ToList();
                var num = random.Next(0, trainData.Count);
                
                var test = trainData[num];
                
                if (trainData.Count < 10 &&  trainData.Count >= TestCount)
                {
                    test = trainData[i];
                }
                
                var output = Forward(new Batch(1, new List<TrainData>{test}));
                Console.WriteLine("=======TEST========");
                
                PrintCase(test, output[0]);
                Console.WriteLine($"Sample Error: {LossFunction.SampleLossFunction(output[0], test.ExpectedY)}");
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

            return sum / Errors.Count;
        }
    }
}