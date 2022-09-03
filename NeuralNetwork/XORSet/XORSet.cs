using System.Collections.Generic;

namespace NeuralNetwork
{
    public class XORSet
    {

        public static List<XORSample> Set()
        {
            return new List<XORSample>
            {
                new XORSample
                {
                    Input = new double[] {0, 0,},
                    Output = new double[] {0},
                },
                new XORSample
                {
                    Input = new double[] {0, 1,},
                    Output = new double[] {1},
                },
                new XORSample
                {
                    Input = new double[] {1, 0,},
                    Output = new double[] {1},
                },
                new XORSample
                {
                    Input = new double[] {1, 1,},
                    Output = new double[] {0},
                },
            };
        }
        
    }
}