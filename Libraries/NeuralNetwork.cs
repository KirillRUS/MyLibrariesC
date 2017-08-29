using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;


namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public float SpeedOfTraining = 0.7f;
        Random rnd = new Random();
        public Layer[] LayerArray;
        public NeuralNetwork(int NumberOfNeuralLayers, int NumberOfNeuronsInNeuralLayer, int NumberOfInputNeurons, int NumberOfOutputNeurons)
        {
            LayerArray = new Layer[NumberOfNeuralLayers];


            for (int i = 1; i < LayerArray.Length - 1; i++)
                if (i != LayerArray.Length - 2)
                    LayerArray[i] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfNeuronsInNeuralLayer, NumberOfNeuronsInNeuralLayer, rnd);
                else
                    LayerArray[i] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfNeuronsInNeuralLayer, NumberOfOutputNeurons, rnd);
            if (NumberOfNeuralLayers > 2)
                LayerArray[0] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfInputNeurons, NumberOfNeuronsInNeuralLayer, rnd);
            else
                LayerArray[0] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfInputNeurons, NumberOfOutputNeurons, rnd);
            LayerArray[LayerArray.Length - 1] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfOutputNeurons, 0, rnd);



        }
        public float[] Calculate(float[] Input)
        {
            NeuralLayer[] LayerArray = new NeuralLayer[this.LayerArray.Length];
            for (int i = 0; i < this.LayerArray.Length; i++)
                LayerArray[i] = (NeuralLayer)this.LayerArray[i];

            for (int i = 0; i < LayerArray.Length; i++)
                for (int j = 0; j < LayerArray[i].NeuronArray.Length; j++)
                    LayerArray[i].NeuronArray[j].InputValue = 0;

            for (int i = 0; i < LayerArray[0].NeuronArray.Length; i++)
                LayerArray[0].NeuronArray[i].InputValue = Input[i];

            for (int i = 0; i < LayerArray.Length - 1; i++)
            {
                for (int j = 0; j < LayerArray[i].NeuronArray.Length; j++)
                {
                    LayerArray[i].NeuronArray[j].Calculate(LayerArray[i + 1]);
                }
            }

            float[] Output = new float[LayerArray[LayerArray.Length - 1].NeuronArray.Length];
            for (int i = 0; i < LayerArray[LayerArray.Length - 1].NeuronArray.Length; i++)
                Output[i] = LayerArray[LayerArray.Length - 1].NeuronArray[i].InputValue;

            return Output;
        }
        public float Training(int NEpoch, TrainingSet[] TrainingSetArray)
        {
            float Erorr = 1;
            for (int e = 0; e < NEpoch; e++)
            {
                Erorr = 1;
                for (int i = 0; i < TrainingSetArray.Length; i++)
                {
                    float[] answer = Calculate(TrainingSetArray[i].InputArray);
                    for (int j = 0; j < answer.Length; j++)
                    {
                        ((NeuralLayer)LayerArray[LayerArray.Length - 1]).NeuronArray[j].Error = TrainingSetArray[i].OutputArray[j] - answer[j];
                        Erorr = (float)Math.Pow(TrainingSetArray[i].OutputArray[j] - answer[j], 2);//100 99
                    }
                    for (int j = LayerArray.Length - 2; j >= 0; j--)
                    {
                        for (int k = 0; k < ((NeuralLayer)LayerArray[j]).NeuronArray.Length; k++)
                        {
                            ((NeuralLayer)LayerArray[j]).NeuronArray[k].CalculateDeltaW((NeuralLayer)LayerArray[j + 1], SpeedOfTraining);
                        }
                    }
                    for (int j = LayerArray.Length - 2; j >= 0; j--)
                    {
                        for (int k = 0; k < ((NeuralLayer)LayerArray[j]).NeuronArray.Length; k++)
                        {
                            ((NeuralLayer)LayerArray[j]).NeuronArray[k].ToCorrectSynapseArray();
                        }
                    }


                }
                Erorr /= TrainingSetArray.Length;
                if (e % 100 == 0)
                    Console.WriteLine(Erorr);
                Thread.Sleep(3);
            }
            return Erorr;
        }


    }
    public class ConvolutionaryNeuralNetwork
    {
        public float SpeedOfTraining = 0.7f;

        Random rnd = new Random();
        public Layer[] LayerArray;
        public ConvolutionaryNeuralNetwork(int NumberOfConvolutionaryLayers, int NFiltersByLayer, int SizeOfFilter,
            int NumberOfNeuralLayers, int NumberOfNeuronsInNeuralLayer, int NumberOfInputNeurons, int NumberOfOutputNeurons)
        {
            LayerArray = new Layer[NumberOfNeuralLayers + NumberOfConvolutionaryLayers * 2];

            LayerArray[0] = new CompressionLayer(Layer.LayerType.CompressionLayer);
            LayerArray[1] = new ConvolutionaryLayer(Layer.LayerType.ConvolutionaryLayer, SizeOfFilter, SizeOfFilter, 3, NFiltersByLayer, rnd);
            for (int i = 2; i < NumberOfConvolutionaryLayers * 2; i += 2)
            {
                LayerArray[i] = new CompressionLayer(Layer.LayerType.CompressionLayer);
                LayerArray[i + 1] = new ConvolutionaryLayer(Layer.LayerType.ConvolutionaryLayer, SizeOfFilter, SizeOfFilter, NFiltersByLayer, NFiltersByLayer, rnd);
            }


            for (int i = NumberOfConvolutionaryLayers * 2; i < LayerArray.Length - 1; i++)
                if (i != LayerArray.Length - 2)
                    LayerArray[i] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfNeuronsInNeuralLayer, NumberOfNeuronsInNeuralLayer, rnd);
                else
                    LayerArray[i] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfNeuronsInNeuralLayer, NumberOfOutputNeurons, rnd);

            if (NumberOfNeuralLayers > 2)
                LayerArray[NumberOfConvolutionaryLayers * 2] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfInputNeurons, NumberOfNeuronsInNeuralLayer, rnd);
            else
                LayerArray[NumberOfConvolutionaryLayers * 2] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfInputNeurons, NumberOfOutputNeurons, rnd);
            LayerArray[LayerArray.Length - 1] = new NeuralLayer(Layer.LayerType.NeuralLayer, NumberOfOutputNeurons, 0, rnd);
        }
        public ConvolutionaryNeuralNetwork(byte[] ByteArray)
        {
            DataArray DA = new DataArray(ByteArray);
            LayerArray = new Layer[DA.ReadInt()];
            for (int i = 0; i < LayerArray.Length; i++)
            {
                int LayerType = DA.ReadInt();
                if (LayerType == 0)
                    LayerArray[i] = new NeuralLayer(Layer.LayerType.NeuralLayer, DA);
                if (LayerType == 1)
                    LayerArray[i] = new ConvolutionaryLayer(Layer.LayerType.ConvolutionaryLayer, DA);
                if (LayerType == 2)
                    LayerArray[i] = new CompressionLayer(Layer.LayerType.CompressionLayer);
            }
            ByteArray = Addition(BitConverter.GetBytes(ByteArray.Length), ByteArray);
        }

        public float[] Calculate(Matrix InputMatrix)
        {
            int index = 0;

            while (LayerArray[index]._LayerType != Layer.LayerType.NeuralLayer && index < LayerArray.Length)
            {
                if (LayerArray[index]._LayerType == Layer.LayerType.CompressionLayer)
                {
                    ((CompressionLayer)LayerArray[index]).InputMatrix = InputMatrix;
                    ((CompressionLayer)LayerArray[index]).Calculate();
                    InputMatrix = ((CompressionLayer)LayerArray[index]).OutputMatrix;
                }
                if (LayerArray[index]._LayerType == Layer.LayerType.ConvolutionaryLayer)
                {
                    ((ConvolutionaryLayer)LayerArray[index]).InputMatrix = InputMatrix;
                    ((ConvolutionaryLayer)LayerArray[index]).Calculate();
                    InputMatrix = ((ConvolutionaryLayer)LayerArray[index]).OutputMatrix;
                }
                index++;
            }

            float[] Input = InputMatrix.ToFloatArray();

            for (int i = index; i < LayerArray.Length; i++)
                for (int j = 0; j < ((NeuralLayer)LayerArray[i]).NeuronArray.Length; j++)
                    ((NeuralLayer)LayerArray[i]).NeuronArray[j].InputValue = 0;

            for (int i = 0; i < ((NeuralLayer)LayerArray[index]).NeuronArray.Length; i++)
                ((NeuralLayer)LayerArray[index]).NeuronArray[i].InputValue = Input[i];

            for (int i = index; i < LayerArray.Length - 1; i++)
            {
                for (int j = 0; j < ((NeuralLayer)LayerArray[i]).NeuronArray.Length; j++)
                {
                    ((NeuralLayer)LayerArray[i]).NeuronArray[j].Calculate(((NeuralLayer)LayerArray[i + 1]));
                }
            }

            float[] Output = new float[((NeuralLayer)LayerArray[LayerArray.Length - 1]).NeuronArray.Length];
            for (int i = 0; i < ((NeuralLayer)LayerArray[LayerArray.Length - 1]).NeuronArray.Length; i++)
                Output[i] = ((NeuralLayer)LayerArray[LayerArray.Length - 1]).NeuronArray[i].InputValue;

            return Output;
        }
        public float Training(int NEpoch, ConvolutionaryTrainingSet[] TrainingSetArray)
        {
            float Erorr = 1;
            for (int e = 0; e < NEpoch; e++)
            {
                Erorr = 1;
                for (int i = 0; i < TrainingSetArray.Length; i++)
                {
                    float[] answer = Calculate(TrainingSetArray[i].InputArray);
                    for (int j = 0; j < answer.Length; j++)
                    {
                        ((NeuralLayer)LayerArray[LayerArray.Length - 1]).NeuronArray[j].Error = TrainingSetArray[i].OutputArray[j] - answer[j];
                        Erorr += (float)Math.Pow(TrainingSetArray[i].OutputArray[j] - answer[j], 2);
                    }

                    int indexFirstNeuralLayer = 0;
                    while (LayerArray[indexFirstNeuralLayer]._LayerType != Layer.LayerType.NeuralLayer)
                        indexFirstNeuralLayer++;
                    for (int j = LayerArray.Length - 2; j >= indexFirstNeuralLayer; j--)
                    {
                        for (int k = 0; k < ((NeuralLayer)LayerArray[j]).NeuronArray.Length; k++)
                        {
                            ((NeuralLayer)LayerArray[j]).NeuronArray[k].CalculateDeltaW((NeuralLayer)LayerArray[j + 1], SpeedOfTraining);
                        }
                    }

                    ((NeuralLayer)LayerArray[indexFirstNeuralLayer]).FillErrorArray(((ConvolutionaryLayer)LayerArray[indexFirstNeuralLayer - 1]).OutputMatrix.W,
                        ((ConvolutionaryLayer)LayerArray[indexFirstNeuralLayer - 1]).OutputMatrix.H, ((ConvolutionaryLayer)LayerArray[indexFirstNeuralLayer - 1]).OutputMatrix.D);

                    for (int j = indexFirstNeuralLayer - 1; j >= 0; j--)
                        if (LayerArray[j]._LayerType == Layer.LayerType.ConvolutionaryLayer)
                            ((ConvolutionaryLayer)LayerArray[j]).CalculateDeltaW(LayerArray[j + 1], SpeedOfTraining);
                        else
                            ((CompressionLayer)LayerArray[j]).CalculateDeltaW(LayerArray[j + 1]);

                    for (int j = LayerArray.Length - 2; j > indexFirstNeuralLayer; j--)
                    {
                        for (int k = 0; k < ((NeuralLayer)LayerArray[j]).NeuronArray.Length; k++)
                        {
                            ((NeuralLayer)LayerArray[j]).NeuronArray[k].ToCorrectSynapseArray();
                        }
                    }
                    for (int j = indexFirstNeuralLayer - 1; j >= 0; j -= 2)
                        ((ConvolutionaryLayer)LayerArray[j]).ToCorrectFiltersArray();


                }
                if (SpeedOfTraining > 0.02f)
                    SpeedOfTraining *= 0.98f;
                else
                    SpeedOfTraining *= (1 - SpeedOfTraining / 2);
                Erorr /= TrainingSetArray.Length;
                if (e % 25 == 0)
                    Console.WriteLine(Erorr);
                //Thread.Sleep(3);
            }
            return Erorr;
        }


        public byte[] ToByteArray()
        {
            byte[] ByteArray = new byte[0];
            ByteArray = Addition(BitConverter.GetBytes(LayerArray.Length), ByteArray);
            for (int i = 0; i < LayerArray.Length; i++)
            {
                ByteArray = Addition(ByteArray, LayerArray[i].ToByteArray());
            }
            ByteArray = Addition(BitConverter.GetBytes(ByteArray.Length), ByteArray);
            return ByteArray;
        }
        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }
    }
    public class TrainingSet
    {
        public float[] InputArray;
        public float[] OutputArray;
        public TrainingSet(float[] InputArray, float[] OutputArray)
        {
            this.InputArray = InputArray;
            this.OutputArray = OutputArray;
        }
    }
    public class ConvolutionaryTrainingSet
    {
        public Matrix InputArray;
        public float[] OutputArray;
        public ConvolutionaryTrainingSet(Matrix InputArray, float[] OutputArray)
        {
            this.InputArray = InputArray;
            this.OutputArray = OutputArray;
        }
    }

    public class Neuron
    {
        public float InputValue, Value, Error;
        public float[] SynapseArray, DeltaW;
        public Neuron(float[] SynapseArray)
        {
            InputValue = 0;
            Value = 0;
            this.SynapseArray = SynapseArray;
        }
        public Neuron(DataArray DA)
        {
            InputValue = 0;
            Value = 0;
            SynapseArray = new float[DA.ReadInt()];
            for (int i = 0; i < SynapseArray.Length; i++)
                SynapseArray[i] = DA.ReadFloat();
        }

        public void Calculate(NeuralLayer NextLayer)
        {
            Value = ActivationFunction(InputValue);
            for (int i = 0; i < SynapseArray.Length; i++)
                NextLayer.NeuronArray[i].InputValue += Value * SynapseArray[i];
        }
        public float ActivationFunction(float Input)
        {
            return (float)(Math.Exp(Input * 2) - 1) / (float)(Math.Exp(Input * 2) + 1);
        }
        public float DerivativeActivationFunction(float Input)
        {
            return (float)(Math.Exp(Input) * 4) / (float)(Math.Exp(Input * 2) + 1);
        }
        public void SetRandomSynapseArray(Random rnd)
        {
            for (int i = 0; i < SynapseArray.Length; i++)
                SynapseArray[i] = (float)rnd.NextDouble() - 0.5f;
        }

        public void CalculateDeltaW(NeuralLayer NextLayer, float SpeedOfTraining)
        {
            DeltaW = new float[SynapseArray.Length];
            for (int i = 0; i < SynapseArray.Length; i++)
            {
                DeltaW[i] = SpeedOfTraining * Value * NextLayer.NeuronArray[i].Error;// * DerivativeActivationFunction(InputValue);
            }

            Error = 0;
            for (int i = 0; i < SynapseArray.Length; i++)
            {
                Error += SynapseArray[i] * NextLayer.NeuronArray[i].Error;
            }
        }
        public void ToCorrectSynapseArray()
        {
            for (int i = 0; i < SynapseArray.Length; i++)
            {
                SynapseArray[i] += DeltaW[i];
                DeltaW[i] = 0;
            }
        }


        public byte[] ToByteArray()
        {
            byte[] ByteArray = new byte[0];
            float[] F = SynapseArray;
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(SynapseArray.Length));
            for (int i = 0; i < F.Length; i++)
            {
                ByteArray = Addition(ByteArray, BitConverter.GetBytes(F[i]));
            }
            return ByteArray;
        }
        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }
    }
    public class Layer
    {
        public enum LayerType { NeuralLayer, ConvolutionaryLayer, CompressionLayer };
        public LayerType _LayerType;

        //Только для сверточных сетей
        public Matrix ErrorArray;
        public virtual byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            return ByteArray;
        }
        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }
    }

    public class NeuralLayer : Layer
    {
        public Neuron[] NeuronArray;
        public NeuralLayer(LayerType _LayerType, int NumberOfNeuronsInLayer, int NumberOfNeuronsInNextLayer, Random rnd)
        {
            this._LayerType = _LayerType;
            NeuronArray = new Neuron[NumberOfNeuronsInLayer];
            for (int i = 0; i < NumberOfNeuronsInLayer; i++)
            {
                NeuronArray[i] = new Neuron(new float[NumberOfNeuronsInNextLayer]);
                NeuronArray[i].SetRandomSynapseArray(rnd);
            }
        }
        public NeuralLayer(LayerType _LayerType, DataArray DA)
        {
            this._LayerType = _LayerType;

            NeuronArray = new Neuron[DA.ReadInt()];
            for (int i = 0; i < NeuronArray.Length; i++)
                NeuronArray[i] = new Neuron(DA);
        }
        public void FillErrorArray(int W, int H, int D)
        {
            ErrorArray = new Matrix(W, H, D);
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        ErrorArray.matrix[i][j][k] = NeuronArray[i + j * W + k * W * H].Error;
        }
        public override byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(NeuronArray.Length));
            for (int i = 0; i < NeuronArray.Length; i++)
            {
                ByteArray = Addition(ByteArray, NeuronArray[i].ToByteArray());
            }
            return ByteArray;
        }
    }
    public class ConvolutionaryLayer : Layer
    {
        public Matrix InputMatrix, OutputMatrix;
        public Matrix[] ArrayFilters, DeltaW;
        private int WFilters, HFilters;
        public ConvolutionaryLayer(LayerType _LayerType, int WFilters, int HFilters, int DepthOfInputMatrix, int NFilters, Random rnd)
        {
            this._LayerType = _LayerType;

            ArrayFilters = new Matrix[NFilters];
            for (int i = 0; i < NFilters; i++)
            {
                ArrayFilters[i] = new Matrix(WFilters, HFilters, DepthOfInputMatrix, rnd);
            }
            this.WFilters = WFilters;
            this.HFilters = HFilters;
        }
        public ConvolutionaryLayer(LayerType _LayerType, DataArray DA)
        {
            this._LayerType = _LayerType;

            ArrayFilters = new Matrix[DA.ReadInt()];
            this.WFilters = DA.ReadInt();
            this.HFilters = DA.ReadInt();
            for (int i = 0; i < ArrayFilters.Length; i++)
            {
                ArrayFilters[i] = new Matrix(DA);
            }
        }

        public void Calculate()
        {
            if (InputMatrix == null)
                Console.WriteLine("Error! Нет входящей матрицы");
            OutputMatrix = new Matrix(InputMatrix.W - WFilters, InputMatrix.H - HFilters, ArrayFilters.Length);
            for (int i = 0; i < InputMatrix.W - WFilters; i++)
                for (int j = 0; j < InputMatrix.H - HFilters; j++)
                {
                    Matrix PartOfMatrix = InputMatrix.GetPartOfMatrix(i, j, WFilters, HFilters);
                    for (int k = 0; k < ArrayFilters.Length; k++)
                        OutputMatrix.matrix[i][j][k] = ActivationFunction((PartOfMatrix * ArrayFilters[k]) / ArrayFilters[k].W / ArrayFilters[k].H / ArrayFilters[k].D);
                }
        }
        public float ActivationFunction(float Input)
        {
            return (float)((Math.Exp(Input * 2) - 1) / (Math.Exp(Input * 2) + 1));
        }

        public void CalculateDeltaW(Layer NextLayer, float SpeedOfTraining)
        {
            DeltaW = new Matrix[ArrayFilters.Length];
            for (int i = 0; i < DeltaW.Length; i++)
            {
                DeltaW[i] = new Matrix(WFilters, HFilters, InputMatrix.D);
            }
            for (int i = 0; i < DeltaW.Length; i++)
                for (int j = 0; j < NextLayer.ErrorArray.W; j++)
                    for (int k = 0; k < NextLayer.ErrorArray.H; k++)
                        DeltaW[i] = DeltaW[i] + SpeedOfTraining / ArrayFilters.Length * NextLayer.ErrorArray.matrix[j][k][i] * InputMatrix.GetPartOfMatrix(j, k, WFilters, HFilters);


            ErrorArray = new Matrix(InputMatrix.W, InputMatrix.H, InputMatrix.D);
            for (int i = 0; i < ErrorArray.W; i++)
                for (int j = 0; j < ErrorArray.H; j++)
                    for (int k = 0; k < ErrorArray.D; k++)
                        ErrorArray.matrix[i][j][k] = 0;

            for (int i = 0; i < NextLayer.ErrorArray.W; i++)
                for (int j = 0; j < NextLayer.ErrorArray.H; j++)
                    for (int f = 0; f < NextLayer.ErrorArray.D; f++)
                        ErrorArray.AddPartOfMatrix(i, j, NextLayer.ErrorArray.matrix[i][j][f] * ArrayFilters[f]);
        }
        public void ToCorrectFiltersArray()
        {
            for (int i = 0; i < ArrayFilters.Length; i++)
            {
                ArrayFilters[i] += DeltaW[i];
                DeltaW[i] = new Matrix(ArrayFilters[i].W, ArrayFilters[i].H, ArrayFilters[i].D);
            }
        }

        public override byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(ArrayFilters.Length));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(WFilters));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(HFilters));
            for (int i = 0; i < ArrayFilters.Length; i++)
            {
                ByteArray = Addition(ByteArray, ArrayFilters[i].ToByteArray());
            }
            return ByteArray;
        }
    }
    public class CompressionLayer : Layer
    {
        public Matrix InputMatrix, OutputMatrix;
        public CompressionLayer(LayerType _LayerType)
        {
            this._LayerType = _LayerType;
        }
        public void Calculate()
        {
            if (InputMatrix == null)
                Console.WriteLine("Error! Нет входящей матрицы");
            OutputMatrix = new Matrix(InputMatrix.W / 2 + InputMatrix.W % 2, InputMatrix.H / 2 + InputMatrix.H % 2, InputMatrix.D);
            for (int i = 0; i < OutputMatrix.W; i++)
                for (int j = 0; j < OutputMatrix.H; j++)
                    for (int k = 0; k < InputMatrix.D; k++)
                        OutputMatrix.matrix[i][j][k] = Math.Max(Math.Max(InputMatrix.GetPoint(i * 2, j * 2, k), InputMatrix.GetPoint(i * 2 + 1, j * 2, k)),
                            Math.Max(InputMatrix.GetPoint(i * 2, j * 2 + 1, k), InputMatrix.GetPoint(i * 2 + 1, j * 2 + 1, k)));
        }

        public void CalculateDeltaW(Layer NextLayer)
        {
            ErrorArray = new Matrix(InputMatrix.W, InputMatrix.H, InputMatrix.D);
            for (int i = 0; i < ErrorArray.W; i++)
                for (int j = 0; j < ErrorArray.H; j++)
                    for (int k = 0; k < ErrorArray.D; k++)
                        ErrorArray.matrix[i][j][k] = NextLayer.ErrorArray.matrix[i / 2][j / 2][k];
        }
        public override byte[] ToByteArray()
        {
            byte[] ByteArray = BitConverter.GetBytes(_LayerType.GetHashCode());
            return ByteArray;
        }
    }

    public class Matrix
    {
        public float[][][] matrix;
        public int W, H, D;
        public Matrix(int W, int H, int D)
        {
            matrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                matrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    matrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        matrix[i][j][k] = 0;
                    }
                }
            }
            this.W = W;
            this.H = H;
            this.D = D;
        }
        public Matrix(int W, int H, int D, Random rnd)
        {
            matrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                matrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    matrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        matrix[i][j][k] = (float)rnd.NextDouble();
                    }
                }
            }
            this.W = W;
            this.H = H;
            this.D = D;
        }
        public Matrix(DataArray DA)
        {
            W = DA.ReadInt();
            H = DA.ReadInt();
            D = DA.ReadInt();
            matrix = new float[W][][];
            for (int i = 0; i < W; i++)
            {
                matrix[i] = new float[H][];
                for (int j = 0; j < H; j++)
                {
                    matrix[i][j] = new float[D];
                    for (int k = 0; k < D; k++)
                    {
                        matrix[i][j][k] = DA.ReadFloat();
                    }
                }
            }
        }

        public void BitmapToMatrix(Bitmap _Image)
        {
            matrix = new float[_Image.Width][][];
            for (int i = 0; i < _Image.Width; i++)
            {
                matrix[i] = new float[_Image.Height][];
                for (int j = 0; j < _Image.Height; j++)
                {
                    Color C = _Image.GetPixel(i, j);
                    matrix[i][j] = new float[3];
                    matrix[i][j][0] = 1 - (float)C.R / 255;
                    matrix[i][j][1] = 1 - (float)C.G / 255;
                    matrix[i][j][2] = 1 - (float)C.B / 255;
                }
            }
            W = _Image.Width;
            H = _Image.Height;
            D = 3;
        }
        public Matrix GetPartOfMatrix(int X, int Y, int W, int H)
        {
            Matrix A = new Matrix(W, H, D);
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        A.matrix[i][j][k] = matrix[i + X][j + Y][k];
            return A;
        }
        public void AddPartOfMatrix(int X, int Y, Matrix P)
        {
            if (X < 0 || Y < 0 || X + P.W >= W || Y + P.H >= H || P.D != D)
                Console.WriteLine("Error! Ошибка при добавлении матрицы!");
            for (int i = 0; i < P.W; i++)
                for (int j = 0; j < P.H; j++)
                    for (int k = 0; k < P.D; k++)
                        matrix[i + X][j + Y][k] += P.matrix[i][j][k];
        }
        public float[] ToFloatArray()
        {
            float[] a = new float[W * H * D];
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        a[i + j * W + k * W * H] = matrix[i][j][k];
            return a;
        }

        public static float operator *(Matrix a, Matrix b)
        {
            if (a.W != b.W || a.H != b.H || a.D != b.D)
                Console.WriteLine("Erorr! Невозможно сложить матрицы!");
            float s = 0;
            for (int i = 0; i < a.W; i++)
                for (int j = 0; j < a.H; j++)
                    for (int k = 0; k < a.D; k++)
                        s += a.matrix[i][j][k] * b.matrix[i][j][k];
            return s;
        }
        public static Matrix operator *(float a, Matrix b)
        {
            Matrix s = new Matrix(b.W, b.H, b.D);
            for (int i = 0; i < b.W; i++)
                for (int j = 0; j < b.H; j++)
                    for (int k = 0; k < b.D; k++)
                        s.matrix[i][j][k] = b.matrix[i][j][k] * a;
            return s;
        }
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.W != b.W || a.H != b.H || a.D != b.D)
                Console.WriteLine("Erorr! Невозможно сложить матрицы!");
            Matrix s = new Matrix(a.W, a.H, a.D);
            for (int i = 0; i < a.W; i++)
                for (int j = 0; j < a.H; j++)
                    for (int k = 0; k < a.D; k++)
                        s.matrix[i][j][k] += a.matrix[i][j][k] + b.matrix[i][j][k];
            return s;
        }
        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.W != b.W || a.H != b.H || a.D != b.D)
                Console.WriteLine("Erorr! Невозможно сложить матрицы!");
            Matrix s = new Matrix(a.W, a.H, a.D);
            for (int i = 0; i < a.W; i++)
                for (int j = 0; j < a.H; j++)
                    for (int k = 0; k < a.D; k++)
                        s.matrix[i][j][k] += a.matrix[i][j][k] - b.matrix[i][j][k];
            return s;
        }
        public byte[] ToByteArray()
        {
            byte[] ByteArray = new byte[0];
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(W));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(H));
            ByteArray = Addition(ByteArray, BitConverter.GetBytes(D));
            for (int i = 0; i < W; i++)
                for (int j = 0; j < H; j++)
                    for (int k = 0; k < D; k++)
                        ByteArray = Addition(ByteArray, BitConverter.GetBytes(matrix[i][j][k]));
            return ByteArray;
        }

        public float GetPoint(int X, int Y, int Z)
        {
            if (matrix != null && X >= 0 && Y >= 0 && Z >= 0 && X < W && Y < H && Z < D)
                return matrix[X][Y][Z];
            else
                return 0;
        }
        public static byte[] Addition(byte[] a, byte[] b)
        {
            byte[] s = new byte[a.Length + b.Length];
            int index;
            for (index = 0; index < a.Length; index++)
                s[index] = a[index];
            for (int i = 0; i < b.Length; i++)
                s[i + index] = b[i];
            return s;
        }
    }
    public class DataArray
    {
        byte[] Array;
        public int index;
        public DataArray(byte[] Array)
        {
            this.Array = Array;
        }
        public int ReadInt()
        {
            index += 4;
            return BitConverter.ToInt32(Array, index - 4);
        }
        public float ReadFloat()
        {
            index += 4;
            return BitConverter.ToSingle(Array, index - 4);
        }
        public double ReadDouble()
        {
            index += 8;
            return BitConverter.ToDouble(Array, index - 8);
        }
    }
}
