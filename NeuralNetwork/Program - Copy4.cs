using System;
using System.Linq;

namespace NeuralNetwork
{
    class Program5
    {
        private static double[] x = FillRandomly(InputsCount); //new double[InputsCount]{1, 2};
        static  double expectedY = 0.1;

        //inputs
        private const int InputsCount = 784;
        
        //neurons
        private const int firstLayerNeuronsCount = 16;
        private const int secondLayerNeuronsCount = 3;
        private const int thirdLayerNeuronsCount = 1;
        
        //weights
        //1st layer
        // private const int firstLayerWeightsXdim = firstLayerNeuronsCount;
        // private const int firstLayerWeightsYDim = InputsCount;
        //
        // //2nd layer
        // private const int secondLayerWeightsXdim = secondLayerNeuronsCount;
        // private const int secondLayerWeightsYdim = firstLayerNeuronsCount;
        //
        // //3rd layer
        // private const int thirdLayerWeightsXdim = thirdLayerNeuronsCount;
        // private const int thirdLayerWeightsYdim = secondLayerNeuronsCount;


        private static double[,] w1 = FillRandomly(firstLayerNeuronsCount, InputsCount);


        private static double[,] w2 = FillRandomly(secondLayerNeuronsCount, firstLayerNeuronsCount);

        private static double[] w3 = FillRandomly(secondLayerNeuronsCount);
      

        private static double lambda = 240;


        private static double[] v1 = new double[firstLayerNeuronsCount];
        private static double[] f1 = new double[firstLayerNeuronsCount];
        private static double[] v2 = new double[secondLayerNeuronsCount];
        private static double[] f2 = new double[secondLayerNeuronsCount];
        private static double vOut = 0;
        private static double y = 0;


        static void Main5(string[] args)
        {
            
            double loss = Double.PositiveInfinity;
            var accuracy = 0.001;
            var i = 0;
            var maxIter = 100000;
            Forward();
            while (Math.Abs(loss) > accuracy && i < maxIter)
            {
                Backward(y);
           
                loss = expectedY - y;
                Console.WriteLine($"ForwardResult : {Forward()}");
                i++;
            }
 
            Console.WriteLine($"Loss:           {loss}");
            Console.WriteLine($"Iterations: {i}");
        }

        private static double[,] FillRandomly(int rowCount, int colCount)
        {
            var random = new Random();
            
            var source = new double[rowCount,colCount];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < colCount; j++)
                {
                    source[i, j] = GetRandomWeight(random);
                }
            }

            return source;
        }

        private static double[] FillRandomly(int rowCount)
        {
            var source = new double[rowCount];
            var random = new Random();
            for (int j = 0; j < rowCount; j++)
            {
                source[j] = GetRandomWeight(random);
            }

            return source;
        }
        
        private static double GetRandomWeight(Random random)
        {
            return random.NextDouble() * (random.Next(0, 1) == 0 ? 1 : -1);
        }

        public static double Forward()
        {
            FirstLayerForward();
            FirstLayerActivation();
            SecondLayerForward();
            SecondLayerActivation();
            ThirdLayerForward();
            ThirdLayerActivation();
            
            return y;
        }

        private static void FirstLayerForward()
        {
            var result = new double[firstLayerNeuronsCount];

            for (int i = 0; i < firstLayerNeuronsCount; i++)
            {
                for (int j = 0; j < InputsCount; j++)
                {
                    result[i] += x[j] * w1[i,j];
                }
            }

            v1 = result;
        }

        private static void FirstLayerActivation()
        {
            f1 = v1.Select(ActivationFunc).ToArray();
        }

        private static void SecondLayerForward()
        {
            double[] result = new double[secondLayerNeuronsCount];
            for (int i = 0; i < secondLayerNeuronsCount; i++)
            {
                for (int j = 0; j < firstLayerNeuronsCount; j++)
                {
                    result[i] += f1[j] * w2[i,j];
                }
            }

            v2 = result;
        }

        private static void SecondLayerActivation()
        {
            f2 = v2.Select(ActivationFunc).ToArray();
        }

        private static void ThirdLayerForward()
        {
            double result = 0;
            for (int i = 0; i < secondLayerNeuronsCount; i++)
            {
                result += f2[i] * w3[i];
            }

            vOut = result;
        }

        private static void ThirdLayerActivation()
        {
             y = ActivationFunc(vOut);
        }
        
        //RELU
        private static double ActivationFunc(double val)
        {
            //return val;
            return (Math.Pow(Math.E, val) - Math.Pow(Math.E, -val)) / (Math.Pow(Math.E, val) + Math.Pow(Math.E, -val));

           // return 1 / (1 + Math.Pow(Math.E, -val));
        }

        //RELU'
        private static double ActivationDerivative(double val)
        {
            //return 1;
            return  1 - val * val;
            // return val * (1 - val);
        }



        private static void Backward(double prediction)
        {
            var e =  prediction - expectedY;
            var delta = e * ActivationDerivative(prediction);
            var secondLayerDeltas = ModifyThirdLayerWeights(delta);
            ModifySecondLayerWeights(secondLayerDeltas);
            ModifyFirstLayerWeights(secondLayerDeltas);
        }

        private static double[] ModifyThirdLayerWeights(double lastNeuronDelta)
        {
            double[] secondLayerDeltas = new double[secondLayerNeuronsCount];


            for (var i = 0; i < secondLayerNeuronsCount; i++)
            {
                secondLayerDeltas[i] = lastNeuronDelta * w3[i] * ActivationDerivative(f2[i]);
            }
            
            //modify weights

            for (int i = 0; i < secondLayerNeuronsCount; i++)
            {
                w3[i] = w3[i] - lambda * secondLayerDeltas[i] * f2[i];
            }

            return secondLayerDeltas;
        }

        private static void ModifySecondLayerWeights(double[] secondLayerDeltas)
        {
            for (int i = 0; i < secondLayerNeuronsCount; i++)
            {
                for (int j = 0; j < firstLayerNeuronsCount; j++)
                {
                    w2[i,j] = w2[i,j] - lambda * secondLayerDeltas[i] * f1[j];
                }
            }
        }

        private static void ModifyFirstLayerWeights(double[] secondLayerDeltas)
        {

            var firstLayerDeltas = new double[firstLayerNeuronsCount];
            for (int i = 0; i < firstLayerNeuronsCount; i++)
            {

                for (int j = 0; j < secondLayerNeuronsCount; j++)
                {
                    firstLayerDeltas[i] += secondLayerDeltas[j] * w2[j,i];
                }
                
                firstLayerDeltas[i] = firstLayerDeltas[i] * ActivationDerivative(f1[i]);
            }


            for (int i = 0; i < firstLayerNeuronsCount; i++)
            {
                for (int j = 0; j < InputsCount; j++)
                {
                    w1[i,j] = w1[i,j] - lambda * firstLayerDeltas[i] * x[j];
                }
            }
        }


    }
}