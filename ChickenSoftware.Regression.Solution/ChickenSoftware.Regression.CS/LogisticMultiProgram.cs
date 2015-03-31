using System;
namespace LobbyGuard.Regression.CS
{
  public class LogisticMultiProgram
  {
    public static void ShowData(double[][] data, int numRows,
      int decimals, bool indices)
    {
      int len = data.Length.ToString().Length;
      for (int i = 0; i < numRows; ++i)
      {
        if (indices == true)
          Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < data[i].Length; ++j)
        {
          double v = data[i][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
        Console.WriteLine("");
      }

      if (numRows < data.Length)
      {
        Console.WriteLine(". . .");
        int lastRow = data.Length - 1;
        if (indices == true)
          Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < data[lastRow].Length; ++j)
        {
          double v = data[lastRow][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
      }
      Console.WriteLine("\n");
    }

    public static void ShowVector(double[] vector, int decimals, 
      bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
        Console.Write(vector[i].ToString("F" + decimals) + " ");
      Console.WriteLine("");
      if (newLine == true)
        Console.WriteLine("");
    }

    public static double[][] MakeDummyData(int numFeatures,
      int numClasses, int numRows, int seed)
    {
      Random rnd = new Random(seed); // make random wts and biases
      double[][] wts = new double[numFeatures][];
      for (int i = 0; i < numFeatures; ++i)
        wts[i] = new double[numClasses];
      double hi = 10.0;
      double lo = -10.0;
      for (int i = 0; i < numFeatures; ++i)
        for (int j = 0; j < numClasses; ++j)
          wts[i][j] = (hi - lo) * rnd.NextDouble() + lo;
      double[] biases = new double[numClasses];
      for (int i = 0; i < numClasses; ++i)
        biases[i] = (hi - lo) * rnd.NextDouble() + lo;

      Console.WriteLine("Generating weights are: ");
      ShowData(wts, wts.Length, 2, true);
      Console.WriteLine("Generating biases are: ");
      ShowVector(biases, 2, true);

      double[][] result = new double[numRows][]; // allocate result
      for (int i = 0; i < numRows; ++i)
        result[i] = new double[numFeatures + numClasses];

      for (int i = 0; i < numRows; ++i) // create one row at a time
      {
        double[] x = new double[numFeatures]; // generate random x-values
        for (int j = 0; j < numFeatures; ++j)
          x[j] = (hi - lo) * rnd.NextDouble() + lo;

        double[] y = new double[numClasses]; // computed outputs storage
        for (int j = 0; j < numClasses; ++j) // compute z-values
        {
          for (int f = 0; f < numFeatures; ++f)
            y[j] += x[f] * wts[f][j];
          y[j] += biases[j];
        }

        // determine loc. of max (no need for 1 / 1 + e^-z)
        int maxIndex = 0;
        double maxVal = y[0];
        for (int c = 0; c < numClasses; ++c)
        {
          if (y[c] > maxVal)
          {
            maxVal = y[c];
            maxIndex = c;
          }
        }
        
        for (int c = 0; c < numClasses; ++c) // convert y to 0s or 1s
          if (c == maxIndex)
            y[c] = 1.0;
          else
            y[c] = 0.0;

        int col = 0; // copy x and y into result
        for (int f = 0; f < numFeatures; ++f)
          result[i][col++] = x[f];
        for (int c = 0; c < numClasses; ++c)
          result[i][col++] = y[c];
      }
      return result;
    }

    public static void SplitTrainTest(double[][] allData, double trainPct, 
      int seed, out double[][] trainData, out double[][] testData)
    {
      Random rnd = new Random(seed);
      int totRows = allData.Length;
      int numTrainRows = (int)(totRows * trainPct); // typically 80% 
      int numTestRows = totRows - numTrainRows;
      trainData = new double[numTrainRows][];
      testData = new double[numTestRows][];

      double[][] copy = new double[allData.Length][]; // ref copy of all data
      for (int i = 0; i < copy.Length; ++i)
        copy[i] = allData[i];

      for (int i = 0; i < copy.Length; ++i) // scramble order
      {
        int r = rnd.Next(i, copy.Length); // use Fisher-Yates
        double[] tmp = copy[r];
        copy[r] = copy[i];
        copy[i] = tmp;
      }
      for (int i = 0; i < numTrainRows; ++i)
        trainData[i] = copy[i];

      for (int i = 0; i < numTestRows; ++i)
        testData[i] = copy[i + numTrainRows];
    } // MakeTrainTest

  } // Program

}// ns
