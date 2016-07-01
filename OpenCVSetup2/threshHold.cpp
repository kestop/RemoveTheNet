#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\opencv.hpp>
#include "threshHold.h"

using namespace std;
using namespace cv;

Mat& scan_c(Mat & M, const uchar * const table);

Mat scan_iterator_from_bristol(Mat M);

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cout << "not engouth paras" << endl;
    }

    Mat M, J;
    M = imread(argv[1], IMREAD_COLOR);

    if (M.empty())
    {
        cout << "image failed to read." << endl;
    }


    Vec3b intensity = M.at<Vec3b>(M.rows / 2, M.cols / 2);
    cout << intensity << "<-intensity\n";

    //the table
    uchar table[256];
    for (size_t i = 0; i < 256; i++)
    {
        if (i > 100 && i < 128) { table[i] = 0; }
        else table[i] = 255;
    }
    //method scan_c
    const int times = 100;
    double t = (double)getTickCount();
    for (size_t i = 0; i < times; i++)
    {
        Mat M_clone = M.clone();
        J = scan_c(M_clone, table);

    }
    t = ((double(getTickCount()) - t) / getTickFrequency()) * 1000;//convert to ms
    t /= times;
    cout << "The time consumed in method 1 'scan_c' is " << t
        << "in " << times << " times, unit ms\n";

    //method2 bristol classical iterator
    t = (double)getTickCount();
    for (size_t i = 0; i < times; i++)
    {
        Mat M_clone = M.clone();
        J = scan_iterator_from_bristol(M_clone);

    }
    t = ((double(getTickCount()) - t) / getTickFrequency()) * 1000;//convert to ms
    t /= times;
    cout << "The time consumed in method 2 is " << t
        << "in " << times << " times, unit ms\n";


    string title("threshHold");
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, J);
    waitKey(0);



    return 0;
}

//method 1 scan-c
Mat& scan_c(Mat& M, const uchar* const table)
{
    //accpet only char type matrices
    CV_Assert(M.depth() == CV_8U);

    int channels = M.channels();

    int nRows = M.rows;
    int nCols = M.cols * channels;


    if (M.isContinuous()) {
        cout << "This image is continuous.\n";
        nCols *= nRows;
        nRows = 1;
    }

    int i, j;
    uchar* p;
    for (i = 0; i < nRows; i++)
    {
        p = M.ptr<uchar>(i);
        for (j = 0; j < nCols; j += 3)
        {
            //access the blue channel
            if (p[j] > 200) {
                p[j] = p[j+1] = p[j+2] = 255;
            }
            else {
                p[j] = p[j + 1] = p[j + 2] = 0;
            }

        }
    }

    return M;
}
//method 2 
Mat scan_iterator_from_bristol(Mat M)
{
    // THRESHOLD BY LOOPING THROUGH ALL PIXELS
    for (int i = 0; i<M.rows; i++) {
        for (int j = 0; j<M.cols; j++) {

            uchar pixelBlue = M.at<Vec3b>(i, j)[0];
            uchar pixelGreen = M.at<Vec3b>(i, j)[1];
            uchar pixelRed = M.at<Vec3b>(i, j)[2];

            if (pixelBlue>200) {
                M.at<Vec3b>(i, j)[0] = 255;
                M.at<Vec3b>(i, j)[1] = 255;
                M.at<Vec3b>(i, j)[2] = 255;
            }
            else {
                M.at<Vec3b>(i, j)[0] = 0;
                M.at<Vec3b>(i, j)[1] = 0;
                M.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }

    return M;
}