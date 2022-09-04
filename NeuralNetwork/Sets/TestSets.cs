using System.Collections.Generic;
using System.Linq;
using Accord.Math;
using NeuralNetwork.ImageProcessing;

namespace NeuralNetwork.Sets
{
    public class TestSets
    {
        public static  List<TrainData> GetTest()
        {
            var trainData = new List<TrainData>
            {
                new TrainData(new double[] {0.1, 0.1}, new double[] {0}),
                new TrainData(new double[] {0.1, 1}, new double[] {1}),
                new TrainData(new double[] {1, 0.1}, new double[] {0}),
                new TrainData(new double[] {1, 1}, new double[] {1}),
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
                    trainData.Add(new TrainData(
                        new double[] {sample.X1, sample.X2}, 
                        new double[] {sample.Y / 100.0})
                    );
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
                trainData.Add(new TrainData(xor.Input, xor.Output));
            }

            return trainData;
        }

        public static List<TrainData> ImagesTrain()
        {
            var images = MnistProcessor.ImageDtos();
        
            var inputImages = images.Select(NeuronInputImage.FromDigitImage).ToList();


            var trainData = new List<TrainData>();
            inputImages.Shuffle();
            foreach (var example in inputImages.Take(4))
            {
                trainData.Add(new TrainData(example.NormalizedBytes, example.VectorizedLabel));
            }

            return trainData;
        }
    }
}