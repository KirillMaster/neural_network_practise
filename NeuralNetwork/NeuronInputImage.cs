using System.Collections.Generic;
using System.Linq;
using Accord.Math;

namespace NeuralNetwork
{
    public class NeuronInputImage
    {
        public double[] NormalizedBytes { get; set; }
        public double[] VectorizedLabel { get; set; }
        
        public int Value { get; set; }

        private NeuronInputImage()
        {
            
        }

        public static int FromVectorizedLabel(double[] vectorizedLabel)
        {
            double maxValue = vectorizedLabel.Max();
            int maxIndex = vectorizedLabel.ToList().IndexOf(maxValue);
            return maxIndex;
        }
        
        public static NeuronInputImage FromDigitImage(DigitImage image)
        {
            var normalizedBytes = new List<double>();

            foreach (var row in image.pixels)
            {
                var normalizedPixels = new List<double>();
                foreach (var pixel in row)
                {
                    normalizedPixels.Add(pixel / 255.0);
                }
                
                normalizedBytes.AddRange(normalizedPixels);
            }

            var labelVal = int.Parse(image.label.ToString());

            var vectorizedLabel = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            vectorizedLabel[labelVal] = 1;
            
            return new NeuronInputImage
            {
                NormalizedBytes = normalizedBytes.ToArray(),
                VectorizedLabel = vectorizedLabel,
                Value = labelVal
            };
        }

        public static int NormalizedToValue(double[] vectorizedLabel)
        {
            return vectorizedLabel.ToList().IndexOf(vectorizedLabel.Max()); 
        }
    }
}