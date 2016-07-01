/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - blur.cpp
// TOPIC: basic image blur via convolution with Gaussian Kernel
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace cv;
using namespace std;
void GaussianBlur(
    cv::Mat &input,
    int size,
    cv::Mat &blurredOutput);
void showImage(string windowName, Mat m);

int main(int argc, char** argv)
{
    // LOADING THE IMAGE
    char* imageName = "../../../car1.png";
    Mat image; image = imread(imageName, 1);
    if (image.empty())
    {
        printf(" No image data \n ");  return -1;
    }

    // CONVERT COLOUR, BLUR AND SAVE
    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    // get blur using customized function
    Mat carBlurred;
    GaussianBlur(gray_image, 23, carBlurred);

    //opencv method
    Mat img = gray_image,blurred, sharpened, sharpenedT, lowContrastMask;
    double sigma = 1, threshold = 5, amount = 1;
    for (int i = 0; i < 1; i++)
    {
        cv::GaussianBlur(img, blurred, Size(3, 3), sigma, sigma);
        lowContrastMask = abs(img - blurred) < threshold;//only contrast within threshold is retained , to be 1
        sharpened = img + (img - blurred)*amount;
        img.copyTo(sharpened, lowContrastMask);//only non-zero elements will be copied to output, to reduce noise
        img = sharpened;
    }
    showImage("OpenCV", img);
    //opencv method2
    Mat img1 = gray_image, blurred1, sharpened1, sharpenedT1, lowContrastMask1;
    double sigma1 = 1, threshold1 = 5, amount1 = 1;
    for (int i = 0; i < 6; i++)
    {
        cv::GaussianBlur(img1, blurred1, Size(3, 3), sigma1, sigma1);
        lowContrastMask1 = abs(img1 - blurred1) < threshold1;//only contrast within threshold is retained , to be 1
        sharpened1 = img1 + (img1 - blurred1)*amount1;
        //img.copyTo(sharpened, lowContrastMask);//only non-zero elements will be copied to output
        img1 = sharpened1;
    }
    showImage("OpenCV2", img1);

    //stackflow method
    //Mat sharpened2 = gray_image.clone();
    //Mat blurred2;
    //for (size_t i = 0; i < 4; i++)
    //{
    //cv::GaussianBlur(sharpened2, blurred2, Size(0,0), sigma, sigma);
    //addWeighted(sharpened2, 1.5, blurred2, -0.5, 0, sharpened2);
    //}
    ////wiki method , maybe also slides method
    //Mat HFpart = abs(gray_image - carBlurred);
    //Mat enhancedPart1 = gray_image + (gray_image - carBlurred)*2;
    //Mat enhancedPart2 = gray_image + (gray_image - carBlurred)*3;
    //Mat enhancedPart3 = gray_image + (gray_image - carBlurred)*3.5;
    //Mat enhancedPart4 = gray_image + (gray_image - carBlurred)*6;

    waitKey(0);

    /*imwrite("blur.jpg", carBlurred);
    cout << "imwrite complete!\n";
    */
    return 0;
}

void showImage(string windowName, Mat m)
{
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, m);
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
    // intialise the output using the input
    blurredOutput.create(input.size(), input.type());

    // create the Gaussian kernel in 1D 
    Mat kX = getGaussianKernel(size, -1);
    Mat kY = getGaussianKernel(size, -1);

    // make it 2D multiply one by the transpose of the other
    cv::Mat kernel = kX * kY.t();

    //CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
    //TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

    // we need to create a padded version of the input
    // or there will be border effects
    int kernelRadiusX = (kernel.size[0] - 1) / 2;
    int kernelRadiusY = (kernel.size[1] - 1) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder(input, paddedInput,
        kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
        cv::BORDER_REPLICATE);

    // now we can do the convoltion
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            double sum = 0.0;
            for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
            {
                for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
                {
                    // find the correct indices we are using
                    int imagex = i + m + kernelRadiusX;
                    int imagey = j + n + kernelRadiusY;
                    int kernelx = m + kernelRadiusX;
                    int kernely = n + kernelRadiusY;

                    // get the values from the padded image and the kernel
                    int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
                    double kernalval = kernel.at<double>(kernelx, kernely);

                    // do the multiplication
                    sum += imageval * kernalval;
                }
            }
            // set the output value as the sum of the convolution
            blurredOutput.at<uchar>(i, j) = (uchar)sum;
        }
    }
}