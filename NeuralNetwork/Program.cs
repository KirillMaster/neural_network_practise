using System;
using System.Collections.Generic;
using System.Linq;
using Accord.DataSets;
using NeuralNetwork.ImageProcessing;

namespace NeuralNetwork
{
    class Program
    {
    static void Main(string[] args)
    {
        var images = MnistProcessor.ImageDtos();
        var inputImages = images.Select(NeuronInputImage.FromDigitImage).ToList();
        var net = new Network();


        // net.TrainTest(inputImages);
        net.TrainMultiply();
        //net.Train(inputImages);
       
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