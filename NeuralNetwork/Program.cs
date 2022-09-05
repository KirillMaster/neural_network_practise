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
            
            var neuralNet = new NeuralNet();
            neuralNet.SetTrainData();
            neuralNet.BuildNet();
            neuralNet.Train();
            neuralNet.Test();
        }
    }
}