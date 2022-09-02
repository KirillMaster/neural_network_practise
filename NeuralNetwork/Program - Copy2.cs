using System;

namespace NeuralNetwork
{
    class Program3
    {
     
        
        static void Main3(string[] args)
        {
            double w11 = 1;
            double w12 = 2;
            double w21 = 3;
            double w22 = 4;
            double w13 = 5;
            double w23 = 6;
            

            double x1 = 1;
            double x2 = 2;

            double y = 0.5;

            double predict = 0;
            double delta = 1;

            double speed = 0.01;

            var maxIter = 200;
            var i = 0;
            while (Math.Abs(delta) > 0.0001 && i < maxIter)
            {
                var h1 = H1(x1, w11, x2, w21);
                var h2 = H2(x1, w12, x2, w22);
                
                predict = Predict(h1,h2,w13,w23);
                
                delta = Delta(predict, y);
                var delta21 = Delta21(delta, w13);
                var delta22 = Delta22(delta, w23);

                w13 = w13 - speed *  delta21;
                w23 = w23 - speed *  delta22;
                w11 = w11 - speed * delta21 * x1;
                w12 = w12 - speed * delta22 * x1;
                w21 = w21 - speed * delta21 * x2;
                w22 = w22 - speed * delta22 * x2;
                i++;
                Console.WriteLine($"predict y : {predict}");

                Console.WriteLine($"w11:{w11}, w12:{w12}, w21:{w21}, w22:{w22}, w13:{w13}, w23:{w23}");
            }

            var result = (x1 * (-0.54) + x2 * (-0.09)) * 3.45 + (x1 * 0.14 + x2 * 0.29) * 4.14;
            Console.WriteLine($"Calc result : {result}");
        }
        
        static double  H1(double x1, double w11, double x2, double w21)
        {
            return x1 * w11 + x2 * w21;
        }
        
        static double H2(double x1, double w12, double x2, double w22)
        {
            return x1 * w12 + x2 * w22;
        }

        static double ReLU(double x )
        {
            return x > 0 ? x : 0;
        }

        static double ReLuDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        static double Predict(double h1, double h2, double w13, double w23)
        {
            return h1 * w13 + h2 * w23;
        }

        static double Delta(double predict, double goal)
        {
            return predict - goal;
        }

        static double Delta21(double delta, double w13)
        {
            return delta * w13;
        }

        static double Delta22(double delta, double w23)
        {
            return delta * w23;
        }

        static double Delta11(double delta21, double delta22, double w11, double w12)
        {
            return delta21 * w11 + delta22 * w12;
        }
        
        static double Delta12(double delta21, double delta22, double w21, double w22)
        {
            return delta21 * w21 + delta22 * w22;
        }
    }
}