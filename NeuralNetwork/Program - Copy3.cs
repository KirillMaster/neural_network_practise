using System;
using System.Linq;

namespace NeuralNetwork
{
    class Program4
    {
        static double[] x =  {1, 2};
        static double expectedY = 123;
        static double[][] w1 =
        {                 //w11 w12  
            new double[2]{ 0.1, 0.2},
            new double[2]{ 0.2, 0.4 },
            new double[2]{ 0.5, 0.1 } 
        };

        static double[][] w2 =
        {
            new double[3] {0.1, 0.2, 1},
            new double[3] {0.2, 0.4, 0.2}
        };

        static double[] w3 =
        {
            0.1,
            0.06
        };

        private static double lambda = 0.0001;


        private static double[] v1 = new double[3];
        private static double[] f1 = new double[3];
        private static double[] v2 = new double[2];
        private static double[] f2 = new double[2];
        private static double vOut = 0;
        private static double y = 0;


        static void Main4(string[] args)
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
            var result = new double[3];

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    result[i] += x[j] * w1[i][j];
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
            double[] result = new double[2];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    result[i] += f1[j] * w2[i][j];
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
            for (int i = 0; i < 2; i++)
            {
                result += f2[i] * w3[i];
            }

            vOut = result;
        }

        private static void ThirdLayerActivation()
        {
             y = ActivationFunc(vOut);
        }
        
        private static double ActivationFunc(double val)
        {
            return val;
            //return (Math.Pow(Math.E, val) - Math.Pow(Math.E, -val)) / (Math.Pow(Math.E, val) + Math.Pow(Math.E, -val));

            //return 1 / (1 + Math.Pow(Math.E, -val));
        }

        private static double ActivationDerivative(double val)
        {
            return 1;
            //return  1 - val * val;
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
            double[] secondLayerDeltas = new double[2];


            for (var i = 0; i < 2; i++)
            {
                secondLayerDeltas[i] = lastNeuronDelta * w3[i] * ActivationDerivative(f2[i]);
            }
            
            //modify weights

            for (int i = 0; i < 2; i++)
            {
                w3[i] = w3[i] - lambda * secondLayerDeltas[i] * f2[i];
            }

            return secondLayerDeltas;
        }

        private static void ModifySecondLayerWeights(double[] secondLayerDeltas)
        {
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    w2[i][j] = w2[i][j] - lambda * secondLayerDeltas[i] * f1[j];
                }
            }
        }

        private static void ModifyFirstLayerWeights(double[] secondLayerDeltas)
        {

            var firstLayerDeltas = new double[3];
            for (int i = 0; i < 3; i++)
            {

                for (int j = 0; j < 2; j++)
                {
                    firstLayerDeltas[i] += secondLayerDeltas[j] * w2[j][i];
                }
                
                firstLayerDeltas[i] = firstLayerDeltas[i] * ActivationDerivative(f1[i]);
            }


            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    w1[i][j] = w1[i][j] - lambda * firstLayerDeltas[i] * x[j];
                }
            }
        }


    }
}