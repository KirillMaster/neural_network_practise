using System;

namespace NeuralNetwork
{
    public class RandomHelper
    {
        public static double[,] FillRandomly(int rowCount, int colCount)
        {
            var random = new Random();
            
            var source = new double[rowCount,colCount];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < colCount; j++)
                {
                    source[i, j] = GetRandomWeight(random);//(i + j) / 10.0;//GetRandomWeight(random);//(i + j) / 10.0;//GetRandomWeight(random);
                }
            }

            return source;
        }

        public static double[] FillRandomly(int rowCount)
        {
            var source = new double[rowCount];
            var random = new Random();
            for (int j = 0; j < rowCount; j++)
            {
                source[j] = GetRandomWeight(random);
            }

            return source;
        }
        
        public static double GetRandomWeight(Random random)
        {
            return random.NextDouble() * (random.Next(0, 1) == 0 ? 1 : -1);
        }
    }
}