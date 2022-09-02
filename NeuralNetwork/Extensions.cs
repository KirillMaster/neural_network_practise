using System;

namespace NeuralNetwork
{
    static class RandomExtensions
    {
        public static void Shuffle<T> (T[] array)
        {
            var random = new Random();
            int n = array.Length;
            while (n > 1) 
            {
                int k = random.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}