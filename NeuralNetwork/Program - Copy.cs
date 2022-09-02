using System;

namespace NeuralNetwork
{
    class Program2
    {
     
        
        static void Main2(string[] args)
        {
            double x = 2;
            double y = 4;
            double w = -1000;
            double speed = 0.0001;
            
            double expectedAccuracy = 0.1;
            var maxIter = 1000000;

            var lossStep = LossStep(x, y, w); 
            var absLossStep = Math.Abs(lossStep);
            var i = 0;
            while (absLossStep >= expectedAccuracy && i < maxIter)
            {
                lossStep = LossStep(x, y, w);
                
                w = w - speed * lossStep;

                absLossStep = Math.Abs(lossStep);
                Console.WriteLine($"Loss: {lossStep}, W: {w}");
                i++;
            }
            
            Console.WriteLine($"W : {w}");
            
            Console.WriteLine("Hello World!");
        }

        private static double LossStep(double x, double y, double w)
        {
            return (y - w * x) * (-x);
        }
    }
}