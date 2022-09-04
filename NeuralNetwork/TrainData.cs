using System.Linq;

namespace NeuralNetwork
{
    public class TrainData
    {
        public double[] X { get; set;}
        public double[] ExpectedY { get; set;} 
        
        public double[] NormalizedX { get; }

        public TrainData(double[] x, double[] expectedY)
        {
            X = x;
            ExpectedY = expectedY;
            NormalizedX = Normalize();
        }

        private double[] Normalize()
        {
            var min = X.Min();
            var max = X.Max();

            if ((max - min) < 1E-30)
            {
                return X;
            }
            
            return X.Select(d => (d - min) / (max - min))
                .ToArray();
        }
    }
}