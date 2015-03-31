using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using LobbyGuard.Regression.CS;

namespace ChickenSoftware.Regression.Tests
{
    [TestClass]
    public class CSLogisticMultiTests
    {
        LogisticMulti _lc = null;
        double[][] _trainData;
        double[][] _testData;

        public CSLogisticMultiTests()
        {
            int numFeatures = 4;
            int numClasses = 3;
            int numRows = 1000;
            int seed = 42;
            var data = LogisticMultiProgram.MakeDummyData(numFeatures, numClasses, numRows, seed);
            LogisticMultiProgram.SplitTrainTest(data, 0.80, 7, out _trainData, out _testData);
            _lc = new LogisticMulti(numFeatures, numClasses);

            int maxEpochs = 100;
            double learnRate = 0.01;
            double decay = 0.10;
            _lc.Train(_trainData, maxEpochs, learnRate, decay);
        }

        [TestMethod]
        public void GetWeights_ReturnExpected()
        {
            double[][] bestWts = _lc.GetWeights();
            var expected = 13.939104508387803;
            var actual = bestWts[0][0];
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GetBiases_ReturnExpected()
        {
            double[] bestBiases = _lc.GetBiases();
            var expected = 11.795019237894717;
            var actual = bestBiases[0];
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GetTrainAccuracy_ReturnExpected()
        {
            var expected = 0.92125;
            var actual = _lc.Accuracy(_trainData);
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void GetTestAccuracy_ReturnExpected()
        {
            var expected = 0.895;
            double actual = _lc.Accuracy(_testData);
            Assert.AreEqual(expected, actual);
        }
    }
}
