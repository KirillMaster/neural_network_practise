using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork.ImageProcessing
{
    public class MnistProcessor
    {
        public MnistProcessor()
        {
            
        }


        public static List<DigitImage> ImageDtos()
        {
            byte[][] pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; ++i)
                pixels[i] = new byte[28];

            var brImages = LoadImages();
            var brLabels = LoadLabels();

            var result = new List<DigitImage>();
            
            // each test image
            for (int di = 0; di < 10000; ++di)
            {
                for (int i = 0; i < 28; ++i)
                {
                    for (int j = 0; j < 28; ++j)
                    {
                        byte b = brImages.ReadByte();
                        pixels[i][j] = b;
                    }
                }

                byte lbl = brLabels.ReadByte();
                
                DigitImage dImage =
                    new DigitImage(pixels, lbl);
                result.Add(dImage);
            }

            return result;
        }


        private static BinaryReader LoadImages()
        {
          
            FileStream ifsImages =
                new FileStream(@"DataSet\t10k-images.idx3-ubyte",
                    FileMode.Open); // test images
            
            BinaryReader brImages =
                new BinaryReader(ifsImages);
            
            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32(); 
            int numRows = brImages.ReadInt32(); 
            int numCols = brImages.ReadInt32(); 


            return brImages;
        }

        private static BinaryReader LoadLabels()
        {
            FileStream ifsLabels =
                new FileStream(@"DataSet\t10k-labels.idx1-ubyte",
                    FileMode.Open); // test labels
            
            BinaryReader brLabels =
                new BinaryReader(ifsLabels);
            
            int magic2 = brLabels.ReadInt32(); 
            int numLabels = brLabels.ReadInt32(); 
            
            return brLabels;
        }
        
        
    }
}