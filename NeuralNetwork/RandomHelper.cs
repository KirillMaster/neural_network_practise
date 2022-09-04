using System;

namespace NeuralNetwork
{
    public class RandomHelper
    {
        public static double[,] FillRandomly(int rowCount, int colCount,double startingPoint = 0)
        {
            var random = new Random();
            
            var source = new double[rowCount,colCount];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < colCount; j++)
                {
                    source[i, j] = GetRandomWeight(random); 
                    // + startingPoint;//(i + j) / 10.0;//GetRandomWeight(random);//(i + j) / 10.0;//GetRandomWeight(random);
                }
            }

            return source;
        }

        public static double[] FillRandomly(int rowCount, double startingPoint = 0)
        {
            var source = new double[rowCount];
            var random = new Random();
            for (int j = 0; j < rowCount; j++)
            {
                source[j] = 0; GetRandomWeight(random); //+ startingPoint;
            }

            return source;
        }
        
        public static double GetRandomWeight(Random random)
        {
            return random.NextDouble() / 10.0;
            return (random.NextDouble() * (random.Next(0, 2) == 0 ? 1 : -1)) / 10.0;
        }
    }
}