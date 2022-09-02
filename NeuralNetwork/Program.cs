using System;
using System.IO;
using Accord.DataSets;
using NeuralNetwork.ImageProcessing;

namespace NeuralNetwork
{
    class Program
    {
    static void Main(string[] args)
    {
        var images = MnistProcessor.ImageDtos();
        
        Console.WriteLine(images[0].ToString());
    }


    // static void Main(string[] args)
        // {
        //     var net = new Network();
        //     net.Train();
        //     var mnist = new MNIST();
        //     var dataSet = mnist.Training;
        //
        //     var a = dataSet.Item1;
        //     
        //     Console.Write(dataSet.Item1.Length);
        // }

     


        private static void Read()
        {
            var mnist = new MNIST();
            var a = mnist.Training;
        }


    }
}