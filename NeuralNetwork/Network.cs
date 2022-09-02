using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Math;

namespace NeuralNetwork
{
    public class Network
    {
        
        private static double[] x = FillRandomly(InputsCount); //new double[InputsCount]{1, 2};

        private static double[] expectedY = new double[thirdLayerNeuronsCount] {0.1, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.4};

        //inputs
        private const int InputsCount = 784;
        
        //neurons
        private const int firstLayerNeuronsCount = 128;
        private const int secondLayerNeuronsCount = 10;
        private const int thirdLayerNeuronsCount = 10;


        private static double[,] w1 = FillRandomly(firstLayerNeuronsCount, InputsCount);
        private static double[] bias1 = FillRandomly(firstLayerNeuronsCount);
        
        private static double[,] w2 = FillRandomly(secondLayerNeuronsCount, firstLayerNeuronsCount);
        private static double[] bias2 = FillRandomly(secondLayerNeuronsCount);
        
        private static double[,] w3 = FillRandomly(thirdLayerNeuronsCount, secondLayerNeuronsCount);
        private static double[] bias3 = FillRandomly(thirdLayerNeuronsCount);
      

        private static double lambda = 0.01;


        private static double[] v1 = new double[firstLayerNeuronsCount];
        private static double[] f1 = new double[firstLayerNeuronsCount];
        private static double[] v2 = new double[secondLayerNeuronsCount];
        private static double[] f2 = new double[secondLayerNeuronsCount];
        private static double[] vOut = new double[thirdLayerNeuronsCount];
        private static double[] y = new double[thirdLayerNeuronsCount];
        


        public void Train(List<NeuronInputImage> images)
        {
            RandomExtensions.Shuffle(images.ToArray());
            
            
            var imagesCount = 10000;
            double loss = Double.PositiveInfinity;
            var accuracy = 0.001;
            var i = 0;
            var maxIter = 1000000;

            foreach (var image in images.Take(imagesCount).Where(x => x.Value == 7  || x.Value == 2))
            {
                x = image.NormalizedBytes;
                expectedY = image.VectorizedLabel;
                Forward();
                Backward(y);
                loss = Loss(expectedY, y);
                Console.WriteLine($"Loss:           {loss}");
                Console.WriteLine($"Forward: {Forward()}");
            }
            
            var random = new Random().Next(0, imagesCount-1);
            Console.WriteLine($"Test: {images[random].Value}");
            Console.WriteLine("Prediction: ");
            x = images[random].NormalizedBytes;
            Console.WriteLine(Forward());
            Console.WriteLine(NeuronInputImage.FromVectorizedLabel(y));
        }
        
        
        // Forward();
        // while (Math.Abs(loss) > accuracy &&  i < maxIter)
        // {
        //     x = images[0].NormalizedBytes;
        //     expectedY = images[0].VectorizedLabel;
        //     
        //     Backward(y);
        //
        //     loss = Loss(expectedY, y);
        //     Console.WriteLine($"ForwardResult : {Forward()}");
        //     i++;
        // }
        //
        // Console.WriteLine($"Loss:           {loss}");
        // Console.WriteLine($"Iterations: {i}");
        //
        //
        // var test = images.Where(x => x.Value == 7).ToList();
        // var another = test[2];
        // another = images[5];
        //
        //
        // x = another.NormalizedBytes;
        // Console.WriteLine(Forward());


        public void TrainTest()
        {
            double loss = Double.PositiveInfinity;
            var accuracy = 0.001;
            var i = 0;
            var maxIter = 1000;
            Forward();
            while (Math.Abs(loss) > accuracy &&  i < maxIter)
            {
                Backward(y);
           
                loss = Loss(expectedY, y);
                Console.WriteLine($"ForwardResult : {Forward()}");
                i++;
            }
 
            Console.WriteLine($"Loss:           {loss}");
            Console.WriteLine($"Iterations: {i}");
        }
        
        private static double Loss(double[] expectedOutput, double[] prediction)
        {
            double sum = 0;
            for(int i = 0; i < expectedOutput.Length; i ++)
            {
                sum += Math.Abs(expectedOutput[i] - prediction[i]);
            }

            return sum / expectedOutput.Length;
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

        public static string Forward()
        {
            FirstLayerForward();
            FirstLayerActivation();
            SecondLayerForward();
            SecondLayerActivation();
            ThirdLayerForward();
            ThirdLayerActivation();
            
            return string.Join(", ", y.Select(z => string.Format("{0:f2}", z)));
        }

        private static void FirstLayerForward()
        {
            var result = new double[firstLayerNeuronsCount];

            for (int i = 0; i < firstLayerNeuronsCount; i++)
            {
                for (int j = 0; j < InputsCount; j++)
                {
                    result[i] += x[j] * w1[i,j] + bias1[i];
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
                    result[i] += f1[j] * w2[i,j] + bias2[i];
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
            double[] result = new double[thirdLayerNeuronsCount];
            
            for (int i = 0; i < thirdLayerNeuronsCount; i++)
            {
                for (int j = 0; j < secondLayerNeuronsCount; j++)
                {
                    result[i] += f2[j] * w3[i, j] + bias3[i];
                }
            }
           

            vOut = result;
        }

        private static void ThirdLayerActivation()
        {
             y = vOut.Select(val => ActivationSoftMax(val, vOut)).ToArray();
        }


        private static double ActivationSoftMax(double val, double[] allLayer)
        {
            double sum = 0; 
            for (int i = 0; i < allLayer.Length; i++)
            {
                sum += Math.Pow(Math.E, allLayer[i]);
            }

            return Math.Pow(Math.E, val) / sum;
        }

        private static double ActivationSoftMaxDerivative(double val, double[] allLayer)
        {
            var softMax = ActivationSoftMax(val, allLayer);

            return softMax * (1 - softMax);
        }
        
        //RELU
        private static double ActivationFunc(double val)
        {
           // return val;
           // return (Math.Pow(Math.E, val) - Math.Pow(Math.E, -val)) / (Math.Pow(Math.E, val) + Math.Pow(Math.E, -val));

             return 1 / (1 + Math.Pow(Math.E, -val));
        }

        //RELU'
        private static double ActivationDerivative(double val)
        {
            //return 1;
           // return  1 - val * val;
             return val * (1 - val);
        }



        private static void Backward(double[] prediction)
        {
            var e =  OutputErrors(prediction);
            var deltas = OutputDeltas(e, prediction);
            var secondLayerDeltas = ModifyThirdLayerWeights(deltas);
            ModifySecondLayerWeights(secondLayerDeltas);
            ModifyFirstLayerWeights(secondLayerDeltas);
        }


        private static double[] OutputErrors(double[] prediction)
        {
            double[] errors = new double[prediction.Length];
            for (int i = 0; i < prediction.Length; i++)
            {
                errors[i] = prediction[i] - expectedY[i];
            }

            return errors;
        }

        private static double[] OutputDeltas(double[] errors, double[] prediction)
        {
            double[] deltas = new double[errors.Length];
            for (int i = 0; i < errors.Length; i++)
            {
                deltas[i] = errors[i] * ActivationSoftMaxDerivative(prediction[i], prediction);
            }

            return deltas;
        }

        private static double[] ModifyThirdLayerWeights(double[] lastNeuronDeltas)
        {
            double[] secondLayerDeltas = new double[secondLayerNeuronsCount];
            
            //modify weights
            for (int i = 0; i < thirdLayerNeuronsCount; i++)
            {
                for (int j = 0; j < secondLayerNeuronsCount; j++)
                {
                    w3[i,j] = w3[i,j] - lambda * lastNeuronDeltas[i] * f2[j];
                    bias3[i] = bias3[i] - lambda * lastNeuronDeltas[i] * 1;
                } 
            }

            for (int i = 0; i < thirdLayerNeuronsCount; i++)
            {
                for (var j = 0; j < secondLayerNeuronsCount; j++)
                {
                    secondLayerDeltas[j] = lastNeuronDeltas[i] * w3[i, j] * ActivationDerivative(f2[j]);
                }
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
                    bias2[i] = bias2[i] - lambda * secondLayerDeltas[i] * 1;
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
                    bias1[i] = bias1[i] - lambda * firstLayerDeltas[i] * 1;
                }
            }
        }
    }
}