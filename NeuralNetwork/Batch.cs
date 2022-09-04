using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Batch
    {
        public int Size { get; set; }
        public List<TrainData> TrainData { get; set; }
        public double[][] ExpectedYs { get; set; }
        
        public Batch(int size, List<TrainData> data)
        {
            Size = size;
            TrainData = data;
            ExpectedYs = data.Select(x => x.ExpectedY.ToArray()).ToArray();
        }

        public static List<Batch> GetBatches(int size, List<TrainData> allTrainingData)
        {
            return allTrainingData
                .Partition(size)
                .Select(x => new Batch(size, x.ToList()))
                .ToList();
        }
    }
    
}