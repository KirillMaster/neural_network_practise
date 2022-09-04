using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.ImageProcessing;

namespace NeuralNetwork.Sets
{
    public class TestSets
    {
        public static  List<TrainData> GetTest()
        {
            var trainData = new List<TrainData>
            {
                new TrainData
                {
                    X = new double[] {0.5, 2, 0.8},
                    ExpectedY = new double[] {0, 1,0,0,0,0,0}
                },
                new TrainData
                {
                    X = new double[] {2, 0.7, 0.4},
                    ExpectedY = new double[] {0.5, 0.1, 0,0,0,0,0.4}
                },
            };
            return trainData;
        }
        
        public static  NumberMultiplier[] GetMultiplyTable()
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


        public static List<TrainData> GetMultiplyTableTrain()
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

        public static List<TrainData> GetXORTrain()
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

        public static List<TrainData> ImagesTrain()
        {
            var images = MnistProcessor.ImageDtos();
        
            var inputImages = images.Select(NeuronInputImage.FromDigitImage).ToList();


            var trainData = new List<TrainData>();
            foreach (var example in inputImages.Take(2))
            {
                trainData.Add(new TrainData
                {
                    X = example.NormalizedBytes,
                    ExpectedY = example.VectorizedLabel
                });
            }

            return trainData;
        }
    }
}