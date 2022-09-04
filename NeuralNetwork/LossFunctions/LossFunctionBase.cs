namespace NeuralNetwork.LossFunctions
{
    public abstract class LossFunctionBase
    {
        public abstract double SampleLossFunction(double[] outputNeurons, double[] expectedY);
        public virtual double LossFunction(double[][] batchOutputs, double[][] batchExpectedYs)
        {
            double totalError = 0;
            for (int i = 0; i < batchOutputs.Length; i++)
            {
                totalError += SampleLossFunction(batchOutputs[i], batchExpectedYs[i]);
            }

            return totalError;
        }
    }
}