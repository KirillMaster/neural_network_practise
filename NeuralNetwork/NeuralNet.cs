using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Math;
using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.LossFunctions;
using NeuralNetwork.Sets;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        private List<Layer> Layers { get; set; }
        private List<Batch> TrainBatches { get; set; }
        private int BatchSize { get; set; } = 1;
        private double EpochCount { get; set; } = 1000;
        private double Accuracy { get; set; } = 80;
        private double Lambda { get; set; } = 0.01;
        private int TestCount { get; set; } = 4;
        private List<double> Errors { get; set; }
        private int InputsCount { get; set; }
        private int OutputsCount { get; set; }

        private ILossFunction LossFunction { get; set; }
        
        public void BuildNet()
        {
            LossFunction = new CrossEntropyLoss();

            var layers = new List<Layer>()
            {
                new Layer(new ReluActivation(), 180),
                new Layer(new SoftMaxActivation(), OutputsCount)
            };

            SetupLayers(layers);
        }
        
        private double[][] Forward(Batch currentBatch)
        {
            return currentBatch
                .TrainDatas
                .Select(ForwardForOneSample)
                .ToArray();
        }
        
        
        private double[] ForwardForOneSample(TrainData currentTrainPair)
        {
            var currentLayerOutputs = currentTrainPair.NormalizedX;
            
            for (int j = 0; j < Layers.Count; j++)
            {
                currentLayerOutputs = LayerForward(Layers[j], currentLayerOutputs);
            }

            return currentLayerOutputs;
        }

        private double[] LayerForward(Layer layer, double[] previousLayerNeurons)
        {
            layer.SetPreviousLayerOutputs(previousLayerNeurons);
            return layer.Forward();
        }

        private void Backward(double[][] batchOutputs, double[][] batchExpectedYs)
        {
            var localGradients = OutputLayerLocalGradients(batchOutputs, batchExpectedYs);
            for (int k = Layers.Count-1; k >= 0; k--)
            {
                localGradients = LayerBackward(Layers[k], localGradients);
            }
        }

        private double[] LayerBackward(Layer layer, double[] nextLayerLocalGradient)
        {
            layer.SetNextLayerLocalGradient(nextLayerLocalGradient);
            return layer.Backward();
        }
        
        private double[] OutputLayerLocalGradients(double[][] outputNeuronsPerSample, double[][] expectedYPerSample)
        {
            var samplesInBatchCount = outputNeuronsPerSample.Length;
            
            double[] outputLayerLocalGradients = new double[OutputsCount];
            
            for (int outputIndex = 0; outputIndex < OutputsCount; outputIndex++)
            {
                for (int sampleIndex  = 0; sampleIndex < samplesInBatchCount; sampleIndex++)
                {
                    outputLayerLocalGradients[outputIndex] += 
                        LossFunction.LossFunctionDerivative(outputNeuronsPerSample[sampleIndex][outputIndex], expectedYPerSample[sampleIndex][outputIndex]) 
                        * Layers[^1].ActivationFuncDerivative(outputNeuronsPerSample[sampleIndex][outputIndex], outputNeuronsPerSample[sampleIndex]);

                    LocalGradientIsNan(outputLayerLocalGradients[outputIndex]);
                }
            }

            return outputLayerLocalGradients;
        }

        private void LocalGradientIsNan(double val)
        {
            if (Double.IsNaN(val))
            {
                throw new ApplicationException("Output layer delta is NAN");      
            }
        }
        

        private void SetupLayers(List<Layer> layers)
        {
            if (layers.Count == 0)
            {
                return;
            }
            
            var previousLayerOutputsCount = InputsCount;
            
            foreach (var layer in layers)
            {
                layer.SetPreviousLayerNeuronsCount(previousLayerOutputsCount);
                layer.SetLambda(Lambda);
                layer.InitLayer();
                previousLayerOutputsCount = layer.GetNeuronsCount();
            }

            Layers = layers;
        }
        
        
        public void SetTrainData()
        {
            //TrainData = TestSets.GetMultiplyTableTrain();
           //TrainData = TestSets.GetXORTrain();
           //TrainBatches = TestSets.GetTest(BatchSize);
           TrainBatches = TestSets.ImagesTrain(BatchSize);
           
           InputsCount = TrainBatches[0].TrainDatas[0].NormalizedX.Length;
           OutputsCount = TrainBatches[0].TrainDatas[0].ExpectedY.Length;
        }


        private double ImagesNetAccuracy(List<TrainData> imagesTestSet)
        {
            var oneItemBatches = imagesTestSet
                .Select(x => new Batch(1, new List<TrainData> {x})).ToList();

            var rightResults = 0;
            foreach (var batch in oneItemBatches)
            {
                var output = Forward(batch)[0];
                var expected = batch.TrainDatas[0].ExpectedY;

                if (NeuronInputImage.NormalizedToValue(output) == NeuronInputImage.NormalizedToValue(expected))
                {
                    rightResults++;
                }
            }

            return ((double) rightResults / (double) oneItemBatches.Count) * 100;
        }


        public void Train()
        {
            double epochLoss = 10000000000;
            double epochAccuracy = 0;
            int k = 0;
            
            while (k < EpochCount && epochAccuracy < Accuracy)
            {
                TrainBatches.Shuffle();
                TrainBatches.ForEach(x => x.TrainDatas.Shuffle());
                
                Errors = new List<double>();
                
                for (int i = 0; i < TrainBatches.Count; i++)
                {
                    var currentTrainBatch = TrainBatches[i];
                    var output = Forward(currentTrainBatch);
                    Errors.Add(LossFunction.LossFunction(output, currentTrainBatch.ExpectedYs));
                    Backward(output, currentTrainBatch.ExpectedYs);
                }

                Print(epochAccuracy, PrintHelper.EpochLoss(Errors.ToArray()), k);
                k++;
            }
            Console.WriteLine($"!=====================!");
            Console.WriteLine($"Epoch Count: {k}");
        }


        private void Print(double epochAccuracy, double epochLoss, int iteration)
        {
            epochAccuracy = ImagesNetAccuracy(
                TrainBatches.SelectMany(x => x.TrainDatas.Take(3))
                    .ToList());

            if (iteration % 1 == 0)
            {
                PrintHelper.PrintEpochAccuracy(epochAccuracy);
                PrintHelper.PrintEpochLoss(epochLoss);
            }
        }

        public void Test()
        {
            var random = new Random();
            
            for (int i = 0; i < TestCount; i++)
            {
                var trainData = TrainBatches.SelectMany(x => x.TrainDatas).ToList();
                var num = random.Next(0, trainData.Count);
                
                var test = trainData[num];
                
                if (trainData.Count < 10 &&  trainData.Count >= TestCount)
                {
                    test = trainData[i];
                }
                
                var output = Forward(new Batch(1, new List<TrainData>{test}));
                Console.WriteLine("=======TEST========");
                
                PrintHelper.PrintCase(test, output[0]);
                Console.WriteLine($"Sample Error: {LossFunction.SampleLossFunction(output[0], test.ExpectedY)}");
            }
        }


    }
}