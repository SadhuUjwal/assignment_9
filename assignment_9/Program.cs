using System;
using System.Linq;
using System.Collections.Generic;
using System.Windows.Forms;
using System.IO;
using System.Drawing;
using System.Text.RegularExpressions;

using PCALib;

namespace assignment_9
{
    class MathHelper
    {
        //Creates a matrix that contains only ones.
        public static IMatrix ones(int nrow, int ncol)
        {
            Matrix ones = new Matrix(nrow, ncol);
            for (int i = 0; i < nrow; i++)
            {
                for (int j = 0; j < ncol; j++)
                {
                    ones[i, j] = 1;
                }
            }
            return ones;
        }

        public static int[] sortIndex(double[] arr)
        {
            int[] index = Enumerable.Range(0, arr.Length).ToArray();
            Array.Sort(arr, index);
            return (index);
        }
    }

    class KNN
    {
        private int kN; //number of nearest neighbours to use
        private IMatrix Xfit; //data features stored. from These points are the nearest neighbours chosen.
        private IMatrix Yfit; //data class labels. from These points are the nearest neighbours chosen.


        //X: d x n matrix. 	each column contains a d dimensional instance
        //Y: 1 x m matrix. each element is the corresponding class label.
        //k: integer 		number of nearest neighbours to use.
        public void fit(IMatrix X, IMatrix Y, int k = 2)
        {
            kN = k;
            Xfit = X.Clone();
            Yfit = Y.Clone();
        }


        //Transform new data. The KNN must be fitted with fit before using this.
        //x: d x 1 matrix. The features of a single instance 
        public double transformSingleInstance(IMatrix x)
        {
            //Compute the squared euclidean distance to all points in X.
            //To this end, subtract x from each intance and save the results in diff.
            var n = Xfit.Columns;
            var d = Xfit.Rows;
            var diff = Xfit.Subtraction(x.Multiply(MathHelper.ones(1, n)));
            //Console.WriteLine (diff.ToString());
            //Compute the euclidean norm for each column
            double[] distance = new double[n];
            for (int i = 0; i < n; i++)
            {
                var column = diff.Submatrix(0, d - 1, i, i); // the ith column
                var s = column.Transpose().Multiply(column);
                distance[i] = s.Trace;
            }
            //Get the first kN indices which columns are the nearest to x.
            int[] indx = MathHelper.sortIndex(distance).Take(kN).ToArray();

            //Get the labels of the nearest intances.
            IMatrix nearestLabels = Yfit.Submatrix(0, 0, indx);
            //Count the number of occurences of the different labels
            Dictionary<double, double> frequency =
                new Dictionary<double, double>();
            for (int i = 0; i < nearestLabels.Columns; i++)
            {
                var label = nearestLabels[0, i];
                if (frequency.ContainsKey(label))
                {
                    frequency[label] += 1.0;
                }
                else {
                    frequency.Add(label, 1.0);
                }
            }
            //Order it by the number of occurences and return the label that occurs most often
            return frequency.OrderByDescending(i => i.Value).First().Key;
        }

        //Transform new data. The KNN must be fitted with fit before using this.
        //x: d x n matrix. The features of a single instance 
        public IMatrix transform(IMatrix X)
        {
            var n = X.Columns;
            var d = X.Rows;
            IMatrix y = new Matrix(1, n);
            for (int i = 0; i < n; i++)
            {
                Console.Write("\r{0}% ", 100 * ((double)i) / n);
                y[0, i] = transformSingleInstance(X.Submatrix(0, d - 1, i, i));
            }
            Console.WriteLine(" done.");
            return (y);
        }
    }

    class PCA
    {
        public IMatrix mean;
        public IMatrix principleComponents;

        //Fit the PCA to the data in X. Must be done before using it with transform.
        //X: d x n matrix. 	each column contains a d dimensional instance
        //r: integer. 		number of principle components
        public void fit(IMatrix X, int r = 2)
        {
            int d = X.Rows;
            int n = X.Columns;
            IEigenvalueDecomposition decomp;
            r = Math.Min(r, d);
            //Multiply by a vector of 1s and devide by n to get the mean
            mean = X.Multiply(MathHelper.ones(n, 1)).Multiply(1.0 / n);
            //Console.WriteLine (mean.ToString());

            //Subtract the mean form every column. Now the instances in Xmean have zero mean.
            IMatrix XMean = X.Subtraction(mean.Multiply(MathHelper.ones(1, n)));
            //Console.WriteLine (XMean.ToString());


            //Now we want to compute the eigenvectors/values of the covariance matrix
            //XMean.Multiply (XMean.Transpose()). This can be done as below.
            //However, when we have many features, then this is slow. The second
            //appraoch is faster and was suggested.

            //Create the (scaled) convariance matrix
            //IMatrix Cov = XMean.Multiply (XMean.Transpose());
            //Get the eigenvalues/vectors. The eigenvalues are sorted in ascending order
            //decomp = Cov.GetEigenvalueDecomposition();
            //double[] evalues = decomp.RealEigenvalues;
            //Get the eigenvectors with the largest eigenvectors, i.e. the last r eigenvectors
            //principleComponents = decomp.EigenvectorMatrix.Submatrix (0,d-1,d-1-r+1,d-1);

            IMatrix M = XMean.Transpose().Multiply(XMean);
            //Get the eigenvalues/vectors. The eigenvalues are sorted in ascending order
            decomp = M.GetEigenvalueDecomposition();
            double[] evalues = decomp.RealEigenvalues;
            //Get the eigenvectors with the largest eigenvectors, i.e. the last r eigenvectors
            // and multiply them with Xmean (to get the eigenvectors of the covariance matrix
            principleComponents = XMean.Multiply(decomp.EigenvectorMatrix.Submatrix(0, n - 1, n - 1 - r + 1, n - 1));
        }

        //Transform new data. The PCA must be fitted with fit before using this.
        //X: d x n matrix. each column contains a d dimensional instance 
        public IMatrix transform(IMatrix X)
        {
            int d = X.Rows;
            int n = X.Columns;
            //Subtract the mean form every column.
            IMatrix XMean = X.Subtraction(mean.Multiply(MathHelper.ones(1, n)));
            return (principleComponents.Transpose().Multiply(XMean));
        }

    }

    class MainClass
    {
        //Compute accuray by comparing the predicted labels Yhat and the ground truth Y.
        private static double computeAccuracy(IMatrix Yhat, IMatrix Y)
        {
            var n = Yhat.Columns;
            int truepositive = 0;
            for (int i = 0; i < n; i++)
            {
                truepositive += Convert.ToInt32(Yhat[0, i] == Y[0, i]);
            }
            return (((double)truepositive) / n);
        }


        //Function to test the PCA and KNN.
        public static void TestPCAKNN()
        {
            PCA pca = new PCA();
            KNN knn = new KNN();
            Matrix X1 = new Matrix(3, 5);
            X1[0, 0] = 2.0;
            X1[0, 1] = 1.0;
            X1[0, 2] = 2.0;
            X1[0, 3] = 2.0;
            X1[0, 4] = 1.0;
            X1[1, 0] = 1.0;
            X1[1, 1] = 4.0;
            X1[1, 2] = 0.0;
            X1[1, 3] = 4.0;
            X1[1, 4] = 1.0;
            X1[2, 0] = 2.0;
            X1[2, 1] = 0.0;
            X1[2, 2] = 8.0;
            X1[2, 3] = 1.0;
            X1[2, 4] = 1.0;
            Matrix X2 = new Matrix(3, 3);
            X2[0, 0] = 2.0;
            X2[0, 1] = 1.2;
            X2[0, 2] = 0.0;
            X2[1, 0] = 1.0;
            X2[1, 1] = 3.0;
            X2[1, 2] = 0.0;
            X2[2, 0] = 2.0;
            X2[2, 1] = 0.0;
            X2[2, 2] = 0.0;
            Matrix Y1 = new Matrix(1, 5);
            Y1[0, 0] = 1;
            Y1[0, 1] = 2;
            Y1[0, 2] = 3;
            Y1[0, 3] = 1;
            Y1[0, 4] = 5;

            Console.WriteLine("X1 = ");
            Console.WriteLine(X1.ToString());
            Console.WriteLine("X2 = ");
            Console.WriteLine(X2.ToString());

            pca.fit(X1);

            var Z1 = pca.transform(X1);
            var Z2 = pca.transform(X2);

            Console.WriteLine("Z1 = ");
            Console.WriteLine(Z1.ToString());
            Console.WriteLine("Z2 = ");
            Console.WriteLine(Z2.ToString());

            knn.fit(X1, Y1, 3);
            var res = knn.transformSingleInstance(X2.Submatrix(0, 2, 0, 0));
            Console.WriteLine(res.ToString());

            var res2 = knn.transform(X2);
            Console.WriteLine(res2.ToString());
        }

        public static void TestReader()
        {
            ImageReader ireader;
            IMatrix X = new Matrix(0, 0);
            IMatrix Y = new Matrix(0, 0);

            FolderBrowserDialog fbd = new FolderBrowserDialog();
            if (fbd.ShowDialog() != DialogResult.OK)
                return;
            ireader = new ImageReader(fbd.SelectedPath, new ConcatGrayValueExtracor());
            ireader.transform(out X, out Y);
            Console.WriteLine(X);
            Console.WriteLine(Y);
        }

        public static void run()
        {
            //Open the training data.
            ImageReader traningImagereader;
            ImageReader testImagereader;
            IMatrix X = new Matrix(0, 0);
            IMatrix Y = new Matrix(0, 0);
            PCA pca = new PCA(); //The principle component model for the concaternated gray values
            KNN knnPca = new KNN(); //The KNN model trained on the data after the PCA
            KNN knnLBP = new KNN(); //The KNN model trained on the LBP features.

            int kNNPCA = 3; //Number of nearest neighbours for PCA
            int kNNLBP = 6; //Number of nearest neighbours for LBP
            int nPC = 10; //Number of principle components
            int LBPRadius = 1; //Radius for LBP

            //***************TRAINING***************
            Console.WriteLine("Please select the directory with the training images.");
            FolderBrowserDialog fbd = new FolderBrowserDialog();
            if (fbd.ShowDialog() != DialogResult.OK)
                return;
            traningImagereader = new ImageReader(fbd.SelectedPath, new ConcatGrayValueExtracor());

            //Train 1.Model: raw gray value pictue + pca + knn
            Console.WriteLine("Train PCA + KNN...");
            traningImagereader.transform(out X, out Y); //Read features into X and labels into Y
            pca.fit(X, nPC); //Train the PCA
            var Z = pca.transform(X); //Transform with the PCA
            knnPca.fit(Z, Y, kNNPCA); //Train the KNN on the transformed values
            Console.WriteLine("");

            //Train 2.Model: LBP featurs + knn
            Console.WriteLine("Train LBP + KNN...");
            traningImagereader.setFeatureExtractor(new LBPFeatureExtractor(LBPRadius)); //Set feature extractor to LBP
            traningImagereader.transform(out X, out Y); //Read features into X and labels into Y
            knnLBP.fit(X, Y, kNNLBP); //Train the KNN on the LBP values
            Console.WriteLine("");


            //***************TESTING***************
            Console.WriteLine("Please select the directory with the test images.");
            fbd = new FolderBrowserDialog();
            if (fbd.ShowDialog() != DialogResult.OK)
                return;
            testImagereader = new ImageReader(fbd.SelectedPath, new ConcatGrayValueExtracor());


            //Evaluate 1. Model: PCA + KNN
            Console.WriteLine("Test PCA + KNN...");
            testImagereader.transform(out X, out Y); //Read features into X and labels into Y
            Z = pca.transform(X);
            var YhatPCA = knnPca.transform(Z);
            var accuracy = computeAccuracy(YhatPCA, Y);
            Console.WriteLine("Accuray of Raw Picture + PCA + KNN: " + accuracy);
            Console.WriteLine("");

            //Evaluate 2.Model: LBP + KNN
            Console.WriteLine("Test LBP + KNN...");
            testImagereader.setFeatureExtractor(new LBPFeatureExtractor(LBPRadius)); //Set feature extractor to LBP
            testImagereader.transform(out X, out Y); //Read features into X and labels into Y
            var YhatLBP = knnLBP.transform(X); //Predict with the KNN from the LBP values
            accuracy = computeAccuracy(YhatLBP, Y);
            Console.WriteLine("Accuray of LBP + KNN: " + accuracy);
            Console.WriteLine("");
        }

        [STAThread]
        public static void Main(string[] args)
        {
            run();
            Console.ReadLine(); 
        }
    }
}
