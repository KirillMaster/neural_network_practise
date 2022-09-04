namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            RunNet();
        }

        static void RunNet()
        {
            
            var lambda = 0.1;
            var epochCount = 10000;
            var accuracy = 0.01;
            
            var neuralNet = new NeuralNet(lambda,epochCount, accuracy);
            neuralNet.SetTrainData();
            neuralNet.Build();
            neuralNet.Train();
            neuralNet.Test();
        }
    }
}