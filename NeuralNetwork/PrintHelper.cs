using System;
using System.Linq;

namespace NeuralNetwork
{
    public static class PrintHelper
    {
        public static void PrintCase(TrainData currentTrainPair, double[] output)
        {
            var input = string.Join(", ", currentTrainPair.X.Select(z => $"{z:f2}"));
            var result = string.Join(", ", output.Select(z => $"{z:f2}"));
            var expected = string.Join(", ", currentTrainPair.ExpectedY.Select(z => $"{z:f2}"));
            Console.WriteLine($"=====================");
            Console.WriteLine($"Output: {result}");
            Console.WriteLine();
            Console.WriteLine($"Expected: {expected}");
        }

        public static void PrintEpochLoss(double loss)
        {
            Console.WriteLine($"EpochLoss: {loss}");
        }

        public static void PrintEpochAccuracy(double accuracy)
        {
            if (accuracy <= 30)
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }

            if (accuracy > 30 && accuracy <= 75)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
            }

            if (accuracy > 75)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }


            Console.WriteLine($"EpochAccuracy {accuracy}%");
            Console.ForegroundColor = ConsoleColor.Gray;
        }

        public static double EpochLoss(double[] errors)
        {
            double sum = 0;
            for (int i = 0; i < errors.Length; i++)
            {
                sum += errors[i];
            }

            return sum / errors.Length;
        }
    }
}