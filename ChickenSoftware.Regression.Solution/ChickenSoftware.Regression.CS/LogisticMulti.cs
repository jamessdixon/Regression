using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LobbyGuard.Regression.CS
{
    public class LogisticMulti
    {
        private int numFeatures;
        private int numClasses;
        private double[][] weights; // [feature][class]
        private double[] biases;    // [class]

        public LogisticMulti(int numFeatures, int numClasses)
        {
            this.numFeatures = numFeatures;
            this.numClasses = numClasses;
            this.weights = MakeMatrix(numFeatures, numClasses);
            this.biases = new double[numClasses];
        }

        private double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        public void SetWeights(double[][] wts, double[] b)
        {
            // set weights[][] and biases[]
            for (int i = 0; i < numFeatures; ++i)
                for (int j = 0; j < numClasses; ++j)
                    this.weights[i][j] = wts[i][j];
            for (int j = 0; j < numClasses; ++j)
                this.biases[j] = b[j];
        }

        public double[][] GetWeights()
        {
            double[][] result = new double[numFeatures][];
            for (int i = 0; i < numFeatures; ++i)
                result[i] = new double[numClasses];
            for (int i = 0; i < numFeatures; ++i)
                for (int j = 0; j < numClasses; ++j)
                    result[i][j] = this.weights[i][j];
            return result;
        }

        public double[] GetBiases()
        {
            double[] result = new double[numClasses];
            for (int j = 0; j < numClasses; ++j)
                result[j] = this.biases[j];
            return result;
        }

        private double[] ComputeOutputs(double[] dataItem)
        {
            // using curr weights[][] and biases[]
            // dataItem can be just x or x+y
            double[] result = new double[numClasses];
            for (int j = 0; j < numClasses; ++j) // compute z
            {
                for (int i = 0; i < numFeatures; ++i)
                    result[j] += dataItem[i] * weights[i][j];
                result[j] += biases[j];
            }

            for (int j = 0; j < numClasses; ++j) // 1 / 1 + e^-z
                result[j] = 1.0 / (1.0 + Math.Exp(-result[j]));

            double sum = 0.0; // softmax scaling
            for (int j = 0; j < numClasses; ++j)
                sum += result[j];

            for (int j = 0; j < numClasses; ++j)
                result[j] = result[j] / sum;

            return result;
        } // ComputeOutputs

        public void Train(double[][] trainData, int maxEpochs,
          double learnRate, double decay)
        {
            // 'batch' approach (aggregate gradients using all data)
            double[] targets = new double[numClasses];
            int msgInterval = maxEpochs / 10;
            int epoch = 0;
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % msgInterval == 0 && epoch != maxEpochs)
                {
                    double mse = Error(trainData);
                    Console.Write("epoch = " + epoch);
                    Console.Write("  error = " + mse.ToString("F4"));
                    double acc = Accuracy(trainData);
                    Console.WriteLine("  accuracy = " + acc.ToString("F4"));
                }

                double[][] weightGrads = MakeMatrix(numFeatures, numClasses);
                double[] biasGrads = new double[numClasses];

                // compute all weight gradients, (all classes, all inputs)
                for (int j = 0; j < numClasses; ++j)
                {
                    for (int i = 0; i < numFeatures; ++i)
                    {
                        for (int r = 0; r < trainData.Length; ++r)
                        {
                            double[] outputs = ComputeOutputs(trainData[r]);
                            for (int k = 0; k < numClasses; ++k)
                                targets[k] = trainData[r][numFeatures + k];
                            double input = trainData[r][i];
                            weightGrads[i][j] += (targets[j] - outputs[j]) * input;
                        }
                    }
                }

                // compute all bias gradients (all classes, all inputs)
                for (int j = 0; j < numClasses; ++j)
                {
                    for (int i = 0; i < numFeatures; ++i)
                    {
                        for (int r = 0; r < trainData.Length; ++r)
                        {
                            double[] outputs = ComputeOutputs(trainData[r]);
                            for (int k = 0; k < numClasses; ++k)
                                targets[k] = trainData[r][numFeatures + k];
                            double input = 1; // 1 is a dummy input
                            biasGrads[j] += (targets[j] - outputs[j]) * input;
                        }
                    }
                }

                // update all weights
                for (int i = 0; i < numFeatures; ++i)
                {
                    for (int j = 0; j < numClasses; ++j)
                    {
                        weights[i][j] += learnRate * weightGrads[i][j];
                        weights[i][j] *= (1 - decay);  // wt decay
                    }
                }

                // update all biases
                for (int j = 0; j < numClasses; ++j)
                {
                    biases[j] += learnRate * biasGrads[j];
                    biases[j] *= (1 - decay);
                }

            } // while
        } // Train


        public double Error(double[][] trainData)
        {
            // mean squared error
            double sumSquaredError = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // each training item
            {
                double[] outputs = this.ComputeOutputs(trainData[i]);
                for (int j = 0; j < outputs.Length; ++j)
                {
                    int jj = numFeatures + j; // column in trainData to use
                    sumSquaredError += ((outputs[j] - trainData[i][jj]) *
                      (outputs[j] - trainData[i][jj]));
                }
            }
            return sumSquaredError / trainData.Length;
        }

        public double Accuracy(double[][] trainData)
        {
            // using curr wts and biases
            int numCorrect = 0;
            int numWrong = 0;

            for (int i = 0; i < trainData.Length; ++i)
            {
                int[] deps = ComputeDependents(trainData[i]); // ex: [0  1  0]
                double[] targets = new double[numClasses]; // ex: [0.0  1.0  0.0]
                for (int j = 0; j < numClasses; ++j)
                    targets[j] = trainData[i][numFeatures + j];

                int di = MaxIndex(deps);
                int ti = MaxIndex(targets);
                if (di == ti)
                    ++numCorrect;
                else
                    ++numWrong;
            }

            return (numCorrect * 1.0) / (numWrong + numCorrect);
        }

        private static int MaxIndex(double[] vector)
        {
            int maxIndex = 0;
            double maxVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > maxVal)
                {
                    maxVal = vector[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        private static int MaxIndex(int[] vector)
        {
            int maxIndex = 0;
            int maxVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > maxVal)
                {
                    maxVal = vector[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        private int[] ComputeDependents(double[] dataItem)
        {
            double[] outputs = ComputeOutputs(dataItem); // 0.0 to 1.0
            int maxIndex = MaxIndex(outputs);
            int[] result = new int[numClasses]; // [0 0 .. 0]
            result[maxIndex] = 1;
            return result;
        }

    }
}
