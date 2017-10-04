using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace NeuralNetworkLibrary
{
    class HOG
    {
        public Matrix HOGMatrix;
        private float[][] BWImage;
        public HOG(Bitmap InBitmap)
        {
            //System.DateTime DT = DateTime.Now;
            BWImage = MakeGray(InBitmap);
            ConvertBWImageToHOG();
            //Console.WriteLine("Time: {0}s", (DateTime.Now - DT).TotalSeconds);
        }

        private void ConvertBWImageToHOG()
        {
            HOGMatrix = new Matrix(BWImage.Length / 16, BWImage[0].Length / 16, 8);
            for (int i = 1; i < BWImage.Length - 1; i++)
                for (int j = 1; j < BWImage[i].Length - 1; j++)
                {
                    float x = (BWImage[i + 1][j] - BWImage[i - 1][j]);
                    float y = (BWImage[i][j + 1] - BWImage[i][j - 1]);
                    HOGMatrix.matrix[i / 16][j / 16][GetIndexOfBin(x, y)] = (float)Math.Sqrt(x * x + y * y);
                }
            for (int i = 0; i < HOGMatrix.W; i++)
                for (int j = 0; j < HOGMatrix.H; j++)
                {
                    float s = 0;
                    for (int k = 0; k < HOGMatrix.D; k++)
                        s += HOGMatrix.matrix[i][j][k];
                    s = (float)Math.Sqrt(s);
                    if (s != 0)
                        for (int k = 0; k < HOGMatrix.D; k++)
                            HOGMatrix.matrix[i][j][k] = HOGMatrix.matrix[i][j][k] / s;
                }
        }

        private int GetIndexOfBin(float x, float y)
        {
            int a = 0;
            if (y != 0)
                a = (int)(Math.Atan2(x, y) * 8 / Math.PI / 2);
            else
                a = 0;
            if (a < 0)
                a = 8 + a;
            return a;
        }

        private float[][] MakeGray(Bitmap bmp)
        {
            BWImage = new float[bmp.Width][];
            for (int i = 0; i < BWImage.Length; i++)
                BWImage[i] = new float[bmp.Height];

           // Задаём формат Пикселя.
           PixelFormat pxf = PixelFormat.Format24bppRgb;

            // Получаем данные картинки.
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            //Блокируем набор данных изображения в памяти
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, pxf);

            // Получаем адрес первой линии.
            IntPtr ptr = bmpData.Scan0;

            // Задаём массив из Byte и помещаем в него надор данных.
            // int numBytes = bmp.Width * bmp.Height * 3; 
            //На 3 умножаем - поскольку RGB цвет кодируется 3-мя байтами
            //Либо используем вместо Width - Stride
            int numBytes = bmpData.Stride * bmp.Height;
            int widthBytes = bmpData.Stride;
            byte[] rgbValues = new byte[numBytes];

            // Копируем значения в массив.
            Marshal.Copy(ptr, rgbValues, 0, numBytes);

            // Перебираем пикселы по 3 байта на каждый и меняем значения
            for (int counter = 0; counter < rgbValues.Length; counter += 3)
            {
                int ind = counter / 3;
                BWImage[ind / BWImage[0].Length][ind % BWImage[0].Length] = (float)(rgbValues[counter] + rgbValues[counter + 1] + rgbValues[counter + 2]) / 765;

            }
            // Копируем набор данных обратно в изображение
            //Marshal.Copy(rgbValues, 0, ptr, numBytes);

            // Разблокируем набор данных изображения в памяти.
            bmp.UnlockBits(bmpData);

            return BWImage;
        }
    }
}
