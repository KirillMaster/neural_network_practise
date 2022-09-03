using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        
        private static double[] x = RandomHelper.FillRandomly(InputsCount); //new double[InputsCount]{1, 2};

        private static double[] expectedY = new double[thirdLayerNeuronsCount] {0};
 
        //inputs
        private const int InputsCount = 2;
        
        //neurons
        private const int firstLayerNeuronsCount = 3;
        private const int secondLayerNeuronsCount = 1;
        
        private const int thirdLayerNeuronsCount = 1;


        private static double[,] w1 = RandomHelper.FillRandomly(InputsCount, firstLayerNeuronsCount);
        private static double[] bias1 = RandomHelper.FillRandomly(firstLayerNeuronsCount);
        
        private static double[,] w2 = RandomHelper.FillRandomly(firstLayerNeuronsCount,secondLayerNeuronsCount);
        private static double[] bias2 = RandomHelper.FillRandomly(secondLayerNeuronsCount);
        
        private static double[,] w3 = RandomHelper.FillRandomly(thirdLayerNeuronsCount, secondLayerNeuronsCount);
        private static double[] bias3 = RandomHelper.FillRandomly(thirdLayerNeuronsCount);
      

        private static double lambda = 0.1;


        private static double[] v1 = new double[firstLayerNeuronsCount];
        private static double[] f1 = new double[firstLayerNeuronsCount];
        private static double[] v2 = new double[secondLayerNeuronsCount];
        private static double[] f2 = new double[secondLayerNeuronsCount];
        private static double[] vOut = new double[thirdLayerNeuronsCount];
        private static double[] y = new double[thirdLayerNeuronsCount];
        private static double NetError = 0;
        
        private static int imagesCount = 10000;

        public void Train(List<NeuronInputImage> images)
        {
            images =  RandomExtensions.Shuffle(images.ToArray());
            
            
            double loss = Double.PositiveInfinity;
            var accuracy = 0.001;
            var maxIter = 1000000;

            var batchlength = 100;
            
            
            var batches = Enumerable.Range(0, imagesCount/batchlength)
                .Select(i => images.Skip(i * batchlength)
                    .Take(batchlength)
                    .ToList())
                .ToList();

            foreach (var batch in batches)
            {
                foreach (var image in batch)
                {
                    x = image.NormalizedBytes;
                    expectedY = image.VectorizedLabel;
                    Forward();
                    // Console.WriteLine($"Forward: {Forward()}");
                   // Console.WriteLine($"Loss:           {NetError}");
                }
                
               // Backward(y);
                Console.WriteLine($"Batch Loss:           {NetError}");
            }

            Console.WriteLine($"Loss:           {NetError}");
            var random = new Random().Next(0, imagesCount-1);
            Console.WriteLine($"Test: {images[random].Value}");
            Console.WriteLine("outputNeurons: ");
            x = images[random].NormalizedBytes;
            Console.WriteLine(Forward());
            Console.WriteLine(NeuronInputImage.FromVectorizedLabel(y));
        }


        public void TrainTest()
        {
            var xorSet = XORSet.Set();
            double error = 0;

            var rand = new Random();
            
            for (var epoch = 0; epoch < 20000; epoch++)
            {
                // foreach (var xorSample in xorSet.Take(2))
                // {
                    x = xorSet[0].Input;
                    expectedY = xorSet[0].Output;
                
                    error = Forward();
                    Console.WriteLine($"Loss 1 :           {error}");
                    Backward();
                // }
                
                
                x = xorSet[1].Input;
                expectedY = xorSet[1].Output;
                
                error = Forward();
                Console.WriteLine($"Loss 2 :           {error}");
                Backward();
                
                x = xorSet[2].Input;
                expectedY = xorSet[2].Output;
                
                error = Forward();
                Console.WriteLine($"Loss 3 :           {error}");
                Backward();
                
                x = xorSet[3].Input;
                expectedY = xorSet[3].Output;
                
                error = Forward();
                Console.WriteLine($"Loss 4 :           {error}");
                Backward();
                //Console.WriteLine($"Epoch Loss:         {error}");
            }
            
            
            x = xorSet[0].Input;
            expectedY = xorSet[0].Output;
            Forward();
            Console.WriteLine($"Operation {x[0]} XOR {x[1]}");
            Console.WriteLine($"Result {GetOutputAsString()}");
            
            x = xorSet[1].Input;
            expectedY = xorSet[1].Output;
            Forward();
            Console.WriteLine($"Operation {x[0]} XOR {x[1]}");
            Console.WriteLine($"Result {GetOutputAsString()}");
            
            
            x = xorSet[2].Input;
            expectedY = xorSet[2].Output;
            Forward();
            Console.WriteLine($"Operation {x[0]} XOR {x[1]}");
            Console.WriteLine($"Result {GetOutputAsString()}");
            
            x = xorSet[3].Input;
            expectedY = xorSet[3].Output;
            Forward();
            Console.WriteLine($"Operation {x[0]} XOR {x[1]}");
            Console.WriteLine($"Result {GetOutputAsString()}");
            // var x1 = rand.Next(1, 10);
            // var x2 = rand.Next(1, 10);
            //
            // x = new double[] {x1, x2};
            // expectedY = new double []{x1 * x2};
            // Console.WriteLine($"Operation: {x[0]} * {x[1]}");
            
        }

        public static double Forward()
        {
            FirstLayerForward();
            FirstLayerActivation();
            SecondLayerForward();
            y =SecondLayerActivation();
            // SecondLayerForward();
          
            // ThirdLayerForward();
            // ThirdLayerActivation();
            
            
            return SampleError(y);
        }

        string GetOutputAsString()
        {
            return string.Join(", ", y.Select(z => string.Format("{0:f2}", z)));
        }

        private static void FirstLayerForward()
        {
            var result = new double[firstLayerNeuronsCount];

            for (int i = 0; i < firstLayerNeuronsCount; i++)
            {
                for (int j = 0; j < InputsCount; j++)
                {
                    result[i] += x[j] * w1[j,i] + bias1[i];
                }
            }

            v1 = result;
        }
        
        private static void SecondLayerForward()
        {
            var result = new double[secondLayerNeuronsCount];

            for (int i = 0; i < secondLayerNeuronsCount; i++)
            {
                for (int j = 0; j < firstLayerNeuronsCount; j++)
                {
                    result[i] += f1[j] * w2[j,i] + bias2[i];
                }
            }

            v2 = result;
        }

        private static double[] FirstLayerActivation()
        {
            f1 = v1.Select(ActivationFunc).ToArray();

            //f1 = v1.Select(input => ActivationSoftMax(input, v1)).ToArray();
            return f1;
        }

        

        private static double[] SecondLayerActivation()
        {
            f2 = v2.Select(ActivationFunc).ToArray();
            return f2;
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
            //y = vOut.Select(val => ActivationSoftMax(val, vOut)).ToArray();
            y = vOut.Select(ActivationFunc).ToArray();
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
            //return val >= 0 ? val : 0;
            //return (Math.Pow(Math.E, val) - Math.Pow(Math.E, -val)) / (Math.Pow(Math.E, val) + Math.Pow(Math.E, -val));

             return 1 / (1 + Math.Pow(Math.E, -val));
        }

        //RELU'
        private static double ActivationDerivative(double val)
        {
           // return 1;
            //return  1 - val * val;
            return val * (1 - val);
        }
        

        private static void Backward()
        {
            //var e =  OutputErrors(outputNeurons);

            var deltas = OutputDeltas(y);
            ModifySecondLayerWeights(deltas);

            var firstLayerDeltas = FirstLayerDeltas(deltas);
          //  var secondLayerDeltas = SecondLayerDeltas(deltas);
            ModifyFirstLayerWeights(firstLayerDeltas);
            
            // ModifyThirdLayerWeights(deltas);
            //
            // var secondLayerDeltas = SecondLayerDeltas(deltas);
            // ModifySecondLayerWeights(secondLayerDeltas);
            
          //  var firstLayerDeltas = FirstLayerDeltas(deltas);
       
        }

        private static double SampleError(double[] outputNeurons)
        {
            double error = 0;
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                error += Math.Pow((outputNeurons[i] - expectedY[i]), 2);
            }

            return error;
        }


        private static double CrossEntropyError(double[] outputNeurons)
        {
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                NetError += -1 * expectedY[i] * Math.Log(outputNeurons[i]);
            }

            NetError = NetError / imagesCount;

            return NetError;
        }

        private static double[] OutputDeltas(double[] outputNeurons)
        {
            double[] deltas = new double[outputNeurons.Length];
            for (int i = 0; i < outputNeurons.Length; i++)
            {
                //derivative of loss function
                deltas[i] = 2 * (outputNeurons[i] - expectedY[i]) *ActivationDerivative(outputNeurons[i]);
                            /*ActivationSoftMaxDerivative(outputNeurons[i],
                                outputNeurons);*/ 
            }

            return deltas;
            // double[] deltas = new double[outputNeurons.Length];
            // for (int i = 0; i < outputNeurons.Length; i++)
            // {
            //     deltas[i] = netError * ActivationDerivative(outputNeurons[i]); 
            //     /*ActivationSoftMaxDerivative(outputNeurons[i], outputNeurons);*/
            // }
            //
            // return deltas;
        }

        // private static void ModifyThirdLayerWeights(double[] lastLayerDeltas)
        // {
        //     //modify weights
        //     for (int i = 0; i < secondLayerNeuronsCount; i++)
        //     {
        //         for (int j = 0; j < thirdLayerNeuronsCount; j++)
        //         {
        //             w3[i,j] = w3[i,j] - lambda * lastLayerDeltas[j] * f2[j];
        //             bias3[i] = bias3[i] - lambda * lastLayerDeltas[j] * 1;
        //         } 
        //     }
        // }

        // private static double[] SecondLayerDeltas(double[] lastLayerDeltas)
        // {
        //        
        //     double[] secondLayerDeltas = new double[secondLayerNeuronsCount];
        //     for (int i = 0; i < firstLayerNeuronsCount; i++)
        //     {
        //         for (int j = 0; j < secondLayerNeuronsCount; j++)
        //         {
        //             secondLayerDeltas[i] += lastLayerDeltas[j] * w3[i, j] * ActivationDerivative(f2[j]);
        //         }
        //     }
        //     
        //     return secondLayerDeltas;
        // }

        private static void ModifySecondLayerWeights(double[] outputDeltas)
        {
            for (int i = 0; i < firstLayerNeuronsCount; i++)
            {
                for (int j = 0; j < secondLayerNeuronsCount; j++)
                {
                    w2[i,j] = w2[i,j] - lambda * outputDeltas[j] * f1[i];
                    bias2[j] = bias2[j] - lambda * outputDeltas[j] * 1;
                }
            }
        }

        private static double[] FirstLayerDeltas(double[] outputDeltas)
        {
            var firstLayerDeltas = new double[firstLayerNeuronsCount];

            for (int i = 0; i < firstLayerNeuronsCount; i++)
            {
                for (int j = 0; j < secondLayerNeuronsCount; j++)
                {
                    firstLayerDeltas[i] += outputDeltas[j] * w2[i,j]*  ActivationDerivative(f1[i]);
                }
            }

            return firstLayerDeltas;
        }

        private static void ModifyFirstLayerWeights(double[] firstLayerDeltas)
        {
            for (int i = 0; i < InputsCount; i++)
            {
                for (int j = 0; j < firstLayerNeuronsCount; j++)
                {
                    w1[i,j] = w1[i,j] - lambda * firstLayerDeltas[j] * x[i];
                    bias1[j] = bias1[j] - lambda * firstLayerDeltas[j] * 1;
                }
            }
        }
    }
}