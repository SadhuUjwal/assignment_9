using System;
using System.IO;
using System.Drawing;
using System.Text.RegularExpressions;
using System.Linq;
using System.Collections.Generic;

using PCALib;

namespace assignment_9
{
    //The Image reader reads all the images from a directory and converts each of them
    //to a vector. The vectos are the columnswise concaternated. The class labels are
    //determined by the filename, e.g. S10_3.jpg is the class 10.
    //The injection of the transformation method is handeld by a strategy pattern
    //with the abstract base class ImageFeatureExtractor.
    public class ImageReader
    {
        private String path;
        private ImageFeatureExtractor featureExtractor;

        //Reads the images in the directory path into the matrix X by transforming each image
        //into a vector. Each column corresponds to a image.
        //In Y are the class labels. Y is a vector. Each element corresponds to the class. The 
        //class is determined by the filename of the image. S10_3.jpg is class 10, etc.
        //This image reader reads only the gray values
        public ImageReader(String path, ImageFeatureExtractor featureExtractor)
        {
            this.path = path;
            this.featureExtractor = featureExtractor;
        }

        public void setFeatureExtractor(ImageFeatureExtractor featureExtractor)
        {
            this.featureExtractor = featureExtractor;
        }

        public void transform(out IMatrix X, out IMatrix Y)
        {
            //Read the images and their classes into lists
            List<IMatrix> imageVectors = new List<IMatrix>();
            List<double> imageClasses = new List<double>();
            foreach (var f in Directory.GetFiles(path, "*.jpg"))
            {
                //Console.WriteLine(System.IO.Path.GetFileName(f)); // file name
                Bitmap img = new Bitmap(Image.FromFile(f));
                imageVectors.Add(featureExtractor.transform(img));
                imageClasses.Add(Path2Class(f));
            }
            //Store the results in the matrices X and Y
            var n = imageVectors.Count;

            var d = imageVectors[0].Rows;
            X = new Matrix(d, n);
            Y = new Matrix(1, n);

            for (int i = 0; i < n; i++)
            {
                Y[0, i] = imageClasses[i];
                for (int j = 0; j < d; j++)
                {
                    X[j, i] = imageVectors[i][j, 0];
                }
            }
            Console.WriteLine("Loaded " + n + " images.");
        }

        //Coverts a path like S10_3.jpg into the class 10
        protected double Path2Class(String path)
        {
            double res = 0;
            var regex = new Regex(@"(\d+)(?=_)");
            var filename = System.IO.Path.GetFileName(path);
            var match = regex.Match(filename);
            if (match.Success)
            {
                Double.TryParse(match.Value, out res);
            }
            return res;
        }

    }



    //Base class for extractors that transform images to vectors.
    public abstract class ImageFeatureExtractor
    {
        abstract public IMatrix transform(Bitmap x);
    }

    //Converts Images to vectors by calculating the LBPs
    public class LBPFeatureExtractor : ImageFeatureExtractor
    {

        int radius = 1; //radius for LBP

        public LBPFeatureExtractor(int LBPradius)
        {
            radius = LBPradius;
        }

        /// <summary>
        /// Computes LBP matrix of the image with radius r
        /// </summary>
        /// <param name="srcBmp">source image for which LBP needs to be computed</param>
        /// <param name="r">radius of LBP</param>
        /// <param name="max">max value of the LBP matrix will be returned</param>
        /// <returns>Computed LBP matrix</returns>
        public static double[] LBPMatrix(Bitmap srcBmp, int r, out double max)
        {
            max = 0.0;
            //1. Extract rows and columns from srcImage. Note Source image is Gray scale Converted Image
            int NumRow = srcBmp.Height;
            int numCol = srcBmp.Width;
            List<double> histMat = new List<double>();
            //2. Loop through Pixels
            for (int i = 0; i < NumRow; i++)
                for (int j = 0; j < numCol; j++)
                    //define boundary condition, other wise say if you are looking at pixel (0,0), 
                    //it does not have any suitable neighbors
                    if ((i > r) && (j > r) && (i < (NumRow - r)) && (j < (numCol - r)))
                    {
                        // we want to store binary values in a List
                        List<int> vals = new List<int>();
                        try
                        {
                            for (int i1 = i - r; i1 < (i + r); i1++)
                                for (int j1 = j - r; j1 < (j + r); j1++)
                                {
                                    int acPixel = srcBmp.GetPixel(j, i).R;
                                    int nbrPixel = srcBmp.GetPixel(j1, i1).R;
                                    // 3. This is the main Logic of LBP
                                    if (nbrPixel > acPixel)
                                        vals.Add(1);
                                    else
                                        vals.Add(0);
                                }
                        }
                        catch (Exception ex)
                        {
                        }
                        //4. Once we have a list of 1's and 0's , convert the list to decimal
                        // Also for normalization purpose calculate Max value
                        double dec = ToBinary(vals);
                        histMat.Add(dec);
                        if (dec > max)
                            max = dec;
                    }
            return histMat.ToArray();
        }

        /// <summary>
        /// Convertes list of binary digits to a decimal value
        /// </summary>
        /// <param name="binary">list of binary digits</param>
        /// <returns>decimal value of the list</returns>
        private static double ToBinary(List<int> binary)
        {
            double d = 0;

            for (int i = 0; i < binary.Count; i++)
                d += binary[i] * Math.Pow(2, i);
            return d;
        }

        /// <summary>
        /// computes normalized array of a double array 
        /// </summary>
        /// <param name="Mat">double array which is to be normalized</param>
        /// <param name="max">max value of that array</param>
        /// <returns>normalized array values</returns>
        private static double[] NormalizeLbpMatrix(double[] Mat, double max)
        {
            for (int i = 0; i < Mat.Length; i++)
                Mat[i] = Mat[i] / max;
            return Mat;
        }

        /// <summary>
        /// Provides normalized LBP array of an image
        /// </summary>
        /// <param name="src">source image for which LBP need to e calculated</param>
        /// <param name="r">radius of LBP</param>
        /// <returns></returns>
        private static double[] NormalizedLBP(Bitmap src, int r)
        {
            double max = 0.0;

            var testGray = GrayConversion(src);
            double[] srcMat = NormalizeLbpMatrix(LBPMatrix(testGray, r, out max), max);
            return srcMat;
        }

        /// <summary>
        /// Converts the provided image to gray image
        /// </summary>
        /// <param name="srcBmp">image which needs to be converted as gray image</param>
        /// <returns>gray image</returns>
        public static Bitmap GrayConversion(Bitmap srcBmp)
        {
            int NumRow = srcBmp.Height;
            int numCol = srcBmp.Width;
            Bitmap gray = new Bitmap(srcBmp.Width, srcBmp.Height);

            for (int i = 0; i < NumRow; i++)
                for (int j = 0; j < numCol; j++)
                {
                    // Extract the color of a pixel
                    Color c = srcBmp.GetPixel(j, i);
                    // extract the red,green, blue components from the color.
                    int rd = c.R; int gr = c.G; int bl = c.B;
                    double d1 = 0.2989 * (double)rd + 0.5870 * (double)gr + 0.1140 * (double)bl;
                    int c1 = (int)Math.Round(d1);
                    Color c2 = Color.FromArgb(c1, c1, c1);
                    gray.SetPixel(j, i, c2);
                }
            return gray;
        }

        //Extract from a single image x the LBP.
        public override IMatrix transform(Bitmap x)
        {
            double[] lbp = NormalizedLBP(x, radius);
            IMatrix res = new Matrix(lbp.Length, 1);
            for (int i = 0; i < lbp.Length; i++)
            {
                res[i, 0] = lbp[i];
            }
            return res;
        }
    }

    //Converts a bitmap into a column vector by concaternating all the gray values.
    public class ConcatGrayValueExtracor : ImageFeatureExtractor
    {
        public override IMatrix transform(Bitmap x)
        {
            int NumRow = x.Height;
            int numCol = x.Width;
            IMatrix res = new Matrix(NumRow * numCol, 1);
            for (int i = 0; i < NumRow; i++)
                for (int j = 0; j < numCol; j++)
                {
                    // Extract the color of a pixel
                    Color c = x.GetPixel(j, i);
                    // extract the red,green, blue components from the color.
                    int rd = c.R; int gr = c.G; int bl = c.B;
                    double d1 = 0.2989 * (double)rd + 0.5870 * (double)gr + 0.1140 * (double)bl;
                    int c1 = (int)Math.Round(d1);
                    res[j + i * numCol, 0] = c1;
                }
            return res;
        }
    }
}
