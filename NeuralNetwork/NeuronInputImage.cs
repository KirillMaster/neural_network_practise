using System.Collections.Generic;

namespace NeuralNetwork
{
    public class NeuronInputImage
    {
        public double[] NormalizedBytes { get; set; }
        public int[] VectorizedLabel { get; set; }

        private NeuronInputImage()
        {
            
        }
        
        public static NeuronInputImage FromDigitImage(DigitImage image)
        {
            var normalizedBytes = new List<double>();

            foreach (var row in image.pixels)
            {
                var normalizedPixels = new List<double>();
                foreach (var pixel in row)
                {
                    normalizedPixels.Add(pixel / 255);
                }
                
                normalizedBytes.AddRange(normalizedPixels);
            }

            var labelVal = int.Parse(image.label.ToString());

            var vectorizedLabel = new int[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            vectorizedLabel[labelVal] = 1;
            
            return new NeuronInputImage
            {
                NormalizedBytes = normalizedBytes.ToArray(),
                VectorizedLabel = vectorizedLabel,
            };
        }
    }
}