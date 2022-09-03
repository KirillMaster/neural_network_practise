using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.ImageProcessing;

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

        public void Build()
        {
            Layers = new List<Layer>();
            var inputsCount = TrainData[0].X.Length;
            var outputsCount = TrainData[0].ExpectedY.Length;
            
            var layer1 = new Layer(ActivationFunc, ActivationDerivative, 6, inputsCount, Lambda);
            var layer2 = new Layer(ActivationFunc, ActivationDerivative, outputsCount, layer1.NeuronsCount, Lambda);
            //var layer3 = new Layer(ActivationFunc, ActivationDerivative, 1, layer2.NeuronsCount, Lambda);

            Layers.Add(layer1);
            Layers.Add(layer2);
           // Layers.Add(layer3);
            TestCount = 10;
            Lambda = 0.1;
            Accuracy = 0.0025;
            EpochCount = 10000;
        }

        
        private NumberMultiplier[] GetMultiplyTable()
        {
            var MultiplyTable = new NumberMultiplier[10];
            
            for (int i = 0; i < 10; i++)
            {
                var  numberMultiplier = new NumberMultiplier();
                MultiplyTable[i] = numberMultiplier;

                for (int j = 0; j < 10; j++)
                {
                    var sample = new Sample();
                    numberMultiplier.Samples[j] = sample;
                    sample.X1 = i + 1;
                    sample.X2 = j + 1;
                    sample.Y = sample.X1 * sample.X2;
                }
            }

            return MultiplyTable;
        }


        private List<TrainData> GetMultiplyTableTrain()
        {
              
            var table = GetMultiplyTable();
            var trainData = new List<TrainData>();
            foreach (var number in table)
            {
                foreach (var sample in number.Samples)
                {
                    trainData.Add(new TrainData
                    {
                        X = new double[]{sample.X1, sample.X2},
                        ExpectedY = new double[]{sample.Y/100.0}
                    });
                }
            }

            return trainData;
        }

        private List<TrainData> GetXORTrain()
        {
            var xorSet = XORSet.Set();
            var trainData = new List<TrainData>();
            foreach (var xor in xorSet)
            {
                trainData.Add(new TrainData
                {
                    X = xor.Input,
                    ExpectedY = xor.Output
                });
            }

            return trainData;
        }


        private List<TrainData> GetTest()
        {
            var trainData = new List<TrainData>
            {
                new TrainData
                {
                    X = new double[] {0.5, 2},
                    ExpectedY = new double[] {0, 1}
                },
                new TrainData
                {
                    X = new double[] {2, 0.7},
                    ExpectedY = new double[] {1, 0}
                },
                // new TrainData
                // {
                //     X = new double[] {0.2, 3},
                //     ExpectedY = new double[] {0.6}
                // },
                // new TrainData
                // {
                //     X = new double[] {0.5, 2},
                //     ExpectedY = new double[] {1}
                // },
                // new TrainData
                // {
                //     X = new double[] {0.1, 1},
                //     ExpectedY = new double[] {0.1}
                // },
                // new TrainData
                // {
                //     X = new double[] {0.5, 0.6},
                //     ExpectedY = new double[] {0.3}
                // },
                // new TrainData
                // {
                //     X = new double[] {0.25, 25},
                //     ExpectedY = new double[] {0.625}
                // },
            };
            return trainData;
        }


        private List<TrainData> ImagesTrain()
        {
            var images = MnistProcessor.ImageDtos();
        
            var inputImages = images.Select(NeuronInputImage.FromDigitImage).ToList();


            var trainData = new List<TrainData>();
            foreach (var example in inputImages.Take(100))
            {
                trainData.Add(new TrainData
                {
                    X = example.NormalizedBytes,
                    ExpectedY = example.VectorizedLabel
                });
            }

            return trainData;
        }
        
        public void SetTrainData()
        {
            //TrainData = GetMultiplyTableTrain();
           //TrainData = GetXORTrain();
           TrainData = GetTest();
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
                    Errors.Add(SampleError(output, currentTrainPair.ExpectedY));
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
                Console.WriteLine($"Sample Error: {SampleError(output, test.ExpectedY)}");
            }
        }

        private void PrintCase(TrainData currentTrainPair, double[] output)
        {
            var input = string.Join(", ", currentTrainPair.X.Select(z => $"{z:f2}"));
            var result = string.Join(", ", output.Select(z => $"{z:f2}"));
            var expected = string.Join(", ", currentTrainPair.ExpectedY.Select(z => $"{z:f2}"));
            Console.WriteLine($"=====================");
            Console.WriteLine($"Input: {input}");
            Console.WriteLine($"Output: {result}");
            Console.WriteLine();
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

        //RELU
        private static double ActivationFunc(double val, double[] foo)
        { 
            //return val;
            //return val >= 0 ? val : 0;
            //return (Math.Pow(Math.E, val) - Math.Pow(Math.E, -val)) / (Math.Pow(Math.E, val) + Math.Pow(Math.E, -val));

            return 1 / (1 + Math.Pow(Math.E, -val));
        }

        //RELU'
        private static double ActivationDerivative(double val, double[] foo)
        {
            //return 1;
            //return  1 - val * val;
            return val * (1 - val);
        }
        
        private static double ActivationSoftMax(double val, double[] allLayer)
        {
            double sum = 0; 
            for (int i = 0; i < allLayer.Length; i++)
            {
                sum += Math.Pow(Math.E, allLayer[i]);
            }

            return Math.Pow(Math.E, val) / sum;
        }

        private static double ActivationSoftMaxDerivative(double val, double[] allLayer)
        {
            var softMax = ActivationSoftMax(val, allLayer);

            return softMax * (1 - softMax);
        }
        
        
        private static double[] OutputDeltas(double[] outputNeurons, double[] expectedY)
        {
            double[] deltas = new double[outputNeurons.Length];
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                deltas[i] = 2 * (outputNeurons[i] - expectedY[i]) * ActivationDerivative(outputNeurons[i], outputNeurons);
            }

            return deltas;
        }
    }
}