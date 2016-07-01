#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

void showImage(string windowName, Mat m);
void myMedianBlur(Mat& src, Mat& dst, int ksize);
void insertionSort(int window[]);
int compare(const void *a, const void *b);

int main()
{
    Mat img, img_gray, blurred;
    string name0("../../../car2.png");
    img = imread(name0, IMREAD_COLOR); if (img.empty()) { return -1; }
    cvtColor(img, img_gray, CV_BGR2GRAY);


    medianBlur(img_gray, blurred, 5);
    showImage(name0, blurred);
    //myMedianBlur(img_gray, blurred, 5);


    waitKey(0);
    return 0;
}

void showImage(string windowName, Mat m)
{
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, m);
}

void insertionSort(int window[])
{
    int temp, i, j;
    for (i = 0; i < 9; i++) {
        temp = window[i];
        for (j = i - 1; j >= 0 && temp < window[j]; j--) {
            window[j + 1] = window[j];
        }
        window[j + 1] = temp;
    }
}
int compare(const void *a, const void *b)
{
    return (*(int*)a - *(int*)b);
}

void myMedianBlur(Mat& src, Mat& dst, int ksize)
{
    assert(ksize % 2 == 1);

    int kernelLength = ksize*ksize;
    int * kernel = new int[kernelLength];

    dst = src.clone();
    dst = Scalar(0, 0, 0);
    int count = 0;
    for (int i = ksize; i < src.rows - ksize; i++)
    {
        for (int j = ksize; j < src.cols - ksize; j++)
        {
                count = 0;
            for (int m = 0; m < ksize; m++)
            {
                for (int n = 0; n < ksize; n++)
                {
                    kernel[count] = src.at<uchar>(i - ksize + m, j - ksize + n);
                    count++;
                }
            }
            qsort(kernel, kernelLength, sizeof(int), compare);
            dst.at<uchar>(i, j) = kernel[int(kernelLength / 2)];
        }
    }

    delete[] kernel;

    namedWindow("final");
    imshow("final", dst);

    namedWindow("initial");
    imshow("initial", src);
}