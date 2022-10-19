// CMLPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


using namespace cv;
using namespace std;

Mat RGB2Gray(Mat RGB)
{
    Mat gray = Mat::zeros(RGB.size(), CV_8UC1);

    for (int i = 0; i < RGB.rows; ++i)
    {
	    for (int j = 0; j < RGB.cols * 3; j += 3)
	    {
            gray.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j), RGB.at<uchar>(i, j + 1), RGB.at<uchar>(i, j + 2));
	    }
    }

    return gray;
}

Mat RGB2Binary(Mat RGB, int thr)
{
    Mat gray = RGB2Gray(RGB);

    Mat binary = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 0; i < gray.rows; ++i)
    {
        for (int j = 0; j < gray.cols; ++j)
        {
            if(gray.at<uchar>(i, j) < thr)
            {
                binary.at<uchar>(i, j) = 255;
            }

            //binary.at<uchar>(i, j) = (gray.at<uchar>(i, j) < 127) ? 0 : 255;
            //binary.at<uchar>(i, j) = (std::round(gray.at<uchar>(i, j) / 255.f) * 255.f); //alvins brain
        }
    }
    return binary;
}

Mat Gray2Binary(Mat gray, int thr)
{ 
    Mat binary = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 0; i < gray.rows; ++i)
    {
        for (int j = 0; j < gray.cols; ++j)
        {
            if (gray.at<uchar>(i, j) < thr)
            {
                binary.at<uchar>(i, j) = 0;
            }
            else
            {
                binary.at<uchar>(i, j) = 255;
            }
        }
    }
    return binary;
}

Mat GrayInversion(Mat gray)
{
	Mat InvertedGray = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 0; i < gray.rows; ++i)
    {
        for (int j = 0; j < gray.cols; ++j)
        {
            InvertedGray.at<uchar>(i, j) = 255 - gray.at<uchar>(i, j);
        }
    }
    return InvertedGray;
}

Mat Step(Mat gray, int x, int y)
{
    Mat img = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 0; i < gray.rows; ++i)
    {
        for (int j = 0; j < gray.cols; ++j)
        {
            if (gray.at<uchar>(i, j) > x && gray.at<uchar>(i, j) < y)
            {
                img.at<uchar>(i, j) = 255;
            }
        }
    }
    return img;
}

Mat Masking3x3(Mat gray)
{
    Mat img = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 1; i < gray.rows - 1; ++i)
    {
        for (int j = 1; j < gray.cols - 1; ++j)
        {
            int x = gray.at<uchar>(i - 1, j - 1) + gray.at<uchar>(i - 1, j) + gray.at<uchar>(i - 1, j + 1) +
                gray.at<uchar>(i, j - 1) + gray.at<uchar>(i, j) + gray.at<uchar>(i, j + 1) +
                gray.at<uchar>(i + 1, j - 1) + gray.at<uchar>(i + 1, j) + gray.at<uchar>(i + 1, j + 1);

            x = x / 9;

            img.at<uchar>(i, j) = x;
        }
    }

    return img;
}

Mat Blur(Mat gray, int border )
{
    Mat img = Mat::zeros(gray.size(), CV_8UC1);

    //int totalPix = pow(2 * border + 1, 2);

    for (int i = border; i < gray.rows - border; ++i)
    {
        for (int j = border; j < gray.cols - border; ++j)
        {
            int Total = 0;
            int Count = 0;

            for (int k = i - border; k < i + border; ++k)
            {
	            for (int l = j - border; l < j + border; ++l)
	            {
                    Total += gray.at<uchar>(k, l);
                    Count++;
	            }
            }

            int X = (2 * border + 1) + (2 * border + 1);
            
            Total = Total / X;

            img.at<uchar>(i, j) = Total;
        }
    }

    return img;
}

Mat Max(Mat gray, int border)
{
    Mat img = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = border; i < gray.rows - border; ++i)
    {
        for (int j = border; j < gray.cols - border; ++j)
        {
            int Largest = -1;

            for (int k = i - border; k < i + border; ++k)
            {
                for (int l = j - border; l < j + border; ++l)
                {
                    if(gray.at<uchar>(k, l) > Largest)
                    {
                        Largest = gray.at<uchar>(k, l);
                    }                    
                }
            }            

            img.at<uchar>(i, j) = Largest;
        }
    }

    return img;
}

Mat Min(Mat gray, int border)
{
    Mat img = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = border; i < gray.rows - border; ++i)
    {
        for (int j = border; j < gray.cols - border; ++j)
        {
            int Smallest = 256;

            for (int k = i - border; k < i + border; ++k)
            {
                for (int l = j - border; l < j + border; ++l)
                {
                    if (gray.at<uchar>(k, l) < Smallest)
                    {
                        Smallest = gray.at<uchar>(k, l);
                    }
                }
            }

            img.at<uchar>(i, j) = Smallest;
        }
    }

    return img;
}

Mat Edge(Mat gray, int threshold)
{
    int border = 1;
    Mat img = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = border; i < gray.rows - border; ++i)
    {
        for (int j = border; j < gray.cols - border; ++j)
        {
            int leftAverage = (gray.at<uchar>(i - 1, j - 1) + gray.at<uchar>(i, j - 1) + gray.at<uchar>(i + 1, j - 1)) / 3;
            int RightAverage = (gray.at<uchar>(i - 1, j + 1) + gray.at<uchar>(i, j + 1) + gray.at<uchar>(i + 1, j + 1)) / 3;            

            if(abs(leftAverage - RightAverage) > threshold)
            {
                img.at<uchar>(i, j) = 255;
            }            
        }
    }

    return img;
}

Mat Edge1(Mat Grey, int th)
{
    Mat EdgeImg = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 1; i < Grey.rows - 1; i++)
    {
        for (int j = 1; j < Grey.cols - 1; j++)
        {
            int AvgL = (Grey.at<uchar>(i - 1, j - 1) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i + 1, j - 1)) / 3;
            int AvgR = (Grey.at<uchar>(i - 1, j + 1) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1)) / 3;
            if (abs(AvgL - AvgR) > th)
                EdgeImg.at<uchar>(i, j) = 255;


        }
    }

    return EdgeImg;

}

Mat Dilation(Mat binary, int border)
{
    Mat img = Mat::zeros(binary.size(), CV_8UC1);

    for (int i = border; i < binary.rows - border; ++i)
    {
        for (int j = border; j < binary.cols - border; ++j)
        {
            for (int k = i - border; k < i + border; ++k)
            {
                for (int l = j - border; l < j + border; ++l)
                {
                    if (binary.at<uchar>(k, l) == 255)
                    {
                        img.at<uchar>(i, j) = 255;
                        break;
                    }
                }
            }
        }
    }

    return img;
}

Mat Erosion(Mat binary, int windowsize)
{
    Mat img = Mat::zeros(binary.size(), CV_8UC1);

	for (int i = windowsize; i < binary.rows - windowsize; ++i)
	{
		for (int j = windowsize; j < binary.cols - windowsize; ++j)
		{
			img.at<uchar>(i, j) = binary.at<uchar>(i, j);

			for (int k = i - windowsize; k <= i + windowsize; ++k)
			{
				for (int l = j - windowsize; l <= j + windowsize; ++l)
				{
                    if (binary.at<uchar>(k, l) == 0)
                    {
                        img.at<uchar>(i, j) = 0;
                        break;
                    }
				}
			}
		}
	}
	return img;
}
Mat ErosionOpt(Mat Edge, int windowsize)
{
    Mat ErodedImg = Mat::zeros(Edge.size(), CV_8UC1);

    for (int i = windowsize; i < Edge.rows - windowsize; i++) 
    {
        for (int j = windowsize; j < Edge.cols - windowsize; j++) 
        {
            ErodedImg.at<uchar>(i, j) = Edge.at<uchar>(i, j);
            for (int p = -windowsize; p <= windowsize; p++) 
            {
                for (int q = -windowsize; q <= windowsize; q++) 
                {
                    if (Edge.at<uchar>(i + p, j + q) == 0) {
                        ErodedImg.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    }
    return ErodedImg;
}

Mat Collect(Mat gray, Mat other)
{
    Mat img = Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 0; i < gray.rows; ++i)
    {
        for (int j = 0; j < gray.cols; ++j)
        {

            if (other.at<uchar>(i, j) == 0)
            {
                img.at<uchar>(i, j) = 0;
            }
            else
            {
                img.at<uchar>(i, j) = gray.at<uchar>(i, j);
            }
        }
    }

    return img;
}

void GetCharsFromMat(Mat mat)
{
    
    char* outText;

    tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init("C:\\libraries\\tessdata1", "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    // Open input image with leptonica library
    //Pix* image = pixRead("C:\\images\\1.jpg");
    //api->SetImage(image);

    api->SetImage(mat.data, mat.cols, mat.rows, 1, mat.cols);

    // Get OCR result
    outText = api->GetUTF8Text();
    printf("OCR output:\n%s", outText);

    // Destroy used object and release memory
    api->End();
    delete api;
    delete[] outText;
    //pixDestroy(&image);
}


int main()
{    
    Mat img;
    img = imread("C:\\images\\1.jpg");
    //imshow("RGB image", img);

    Mat GrayImg = RGB2Gray(img);
    //imshow("Gray image", GrayImg);

    //Mat BinaryImg = RGB2Binary(img);
    //imshow("Binary image", BinaryImg);

    //Mat inv = GrayInversion(GrayImg);
    //imshow("inv img", inv);

    //Mat x = Step(GrayImg, 80, 140);
    //imshow("x img", x);
    //
    //Mat masking = Blur(GrayImg, 1);
    //imshow("mask img", masking);

    //Mat max = Max(GrayImg, 1);
    //imshow("max img", max);

    //Mat min = Min(GrayImg, 1);
    //imshow("min img", min);

    //Mat test = Blur(GrayImg, 1);
    //test = Min(test, 1);
    ////test = Step(test, 80, 140);
    //test = Gray2Binary(test);
    //imshow("test img", test);

    //Mat edge = Edge(GrayImg, 50);
    //imshow("Edge img", edge);

    Mat blurEdge = Blur(GrayImg, 1);
    blurEdge = Edge1(blurEdge, 40);
    //blurEdge = Max(blurEdge, 5);
    //imshow("Blur+Edge img", blurEdge);    
        
    Mat ErodedImg = Erosion(blurEdge, 1);
    imshow("erosion", ErodedImg);

	Mat DialatedImg = Dilation(ErodedImg, 14);
    imshow("dilation", DialatedImg);


    Mat DialatedImgCopy;
    DialatedImgCopy = DialatedImg.clone();

    vector<vector<Point>> contours1;
    vector<Vec4i> hierarchy1;
    findContours(DialatedImg, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

    Mat dst = Mat::zeros(GrayImg.size(), CV_8UC3);

    if(!contours1.empty())
    {
	    for (int i = 0; i < contours1.size(); ++i)
	    {
            Scalar color((rand() & 255), (rand() & 255), (rand() & 255));
            drawContours(dst, contours1, i, color, -1, 8, hierarchy1);
	    }
    }

    imshow("segmented img", dst);

    Mat plate;
    Rect rect;
    Scalar black = CV_RGB(0, 0, 0);

    for (int i = 0; i < contours1.size(); ++i)
    {
        rect = boundingRect(contours1[i]);
		float ratio = ((float)rect.width / (float)rect.height);

		if (rect.width < 40 || rect.width > 200 || rect.height > 100 ||
			rect.x < 0.1 * GrayImg.cols || rect.x > 0.9 * GrayImg.cols || rect.y < 0.1 * GrayImg.rows || rect.y > 0.9 * GrayImg.rows ||
			ratio < 1.5)
		{
			drawContours(DialatedImgCopy, contours1, i, black, -1, 8, hierarchy1);
		}
        else
        {
            plate = GrayImg(rect);
        }
	}

    if(plate.rows != 0 || plate.cols != 0)
    {
        imshow("filtered img", DialatedImgCopy);
        plate = Min(plate, 1);
        plate = Gray2Binary(plate, 140);
        imshow("result", plate);

        GetCharsFromMat(plate);
    }


	//Mat collect = Collect(GrayImg, blurEdge);
	//collect = Min(collect, 1);
    //collect = Gray2Binary(collect);



    waitKey();

    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
