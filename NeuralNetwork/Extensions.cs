using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    static class RandomExtensions
    {
        public static List<T> Shuffle<T> (T[] array)
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

            return array.ToList();
        }
        
        public static IEnumerable<IEnumerable<T>> Partition<T>(this IEnumerable<T> values, int chunkSize)
        {
            while (values.Any())
            {
                yield return values.Take(chunkSize).ToList();
                values = values.Skip(chunkSize).ToList();
            }
        }
    }
    
    public class Sample
    {
        public double X1 { get; set; }
        public double X2 { get; set; }
        public double Y { get; set; }
    }

    public class NumberMultiplier
    {
        public NumberMultiplier()
        {
            Samples = new Sample[10];
        }
        public Sample[] Samples { get; set; }
    }
}