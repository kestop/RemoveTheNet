#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\photo.hpp>
#include <opencv2\videoio.hpp>

#include <iostream>
#include <string>
#include <cmath>
#include <ctime>

using namespace cv;
using namespace std;


Mat image0, image, mask;
Mat gray, edges, pyr;
const string videoName("./videos/WillowsSportsCentreCam8.mp4");
const string outputName("./output/sample3.avi");
const string wndName("Image");
const string wndName2("Mask");


static void onMouse(int event, int x, int y, int, void*);
static void onMouseForMaskWnd(int event, int x, int y, int, void*);
static double angle(Point p1, Point p2);
static double length(Point p1, Point p2);
void defineFourEdges(vector<double> angles, double lineLength);
bool ifAtCenter(Point p1, Point p2, const Mat & image);
void findNet(const Mat& image, vector<double> angles, double lineLength);
void makeMask(const vector<Vec4i> lines,
    const vector<double> angles, const double lineLength);
void drawSquare(Mat& src, Mat& dst);
void help();
void eventLoop();
void inpaintWithMask();
void processVideo(const int startFrameIndex);
void Erosion(Mat& src, Mat& erosion_dst);
void Dilation(Mat& src, Mat& dilation_dst);
void capAFrameFromVideo(const int frameIndex);
void maskDisplay();
void findLines(int, void*);

vector<Point> points;
int flag = 'n';
int min_threshold = 50;
int max_trackbar = 150;
int trackbar = max_trackbar/2;
int circleRadius = 2;
bool eraseMode = false;
bool mouseDown = false;
bool maskDisplaying = false;
bool pressDraw = true;//not applied yet
vector<double> angles = { 65,-88,-6,13 }; //default angels and lineLength
double lineLength = 100;

int main(int argc, char* argv[])
{
    help();
    capAFrameFromVideo(0);

    image0.copyTo(image);//Initialize image and mask
    mask = Mat::zeros(image0.size(), CV_8U);

    namedWindow(wndName, WINDOW_AUTOSIZE);//For image display
    namedWindow(wndName2, WINDOW_AUTOSIZE);//For mask display
    setMouseCallback(wndName, onMouse);
    setMouseCallback(wndName2, onMouseForMaskWnd);

    eventLoop();

    return 0;
}
//Get a frame from the video to create mask from 
void capAFrameFromVideo(const int frameIndex)
{
    VideoCapture cap(videoName);
    if (!cap.isOpened())
    {
        cerr << "Video cannot be opened" << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Total frame number is " << cap.get(CV_CAP_PROP_FRAME_COUNT) << "\n"
        << "Current frame is " << cap.get(CAP_PROP_POS_FRAMES) << "\n"
        << "FPS is " << cap.get(CAP_PROP_FPS)
        << endl;
    cap.set(CAP_PROP_POS_FRAMES, frameIndex);
    cap >> image0;
    cout << "Image cols(width) = " << image0.cols << "\n"
        << "Image rows(height) = " << image0.rows
        << endl;

}
void processVideo(const int startFrameIndex)
{
    VideoCapture cap(videoName);
    if (!cap.isOpened())
    {
        cerr << "Video cannot be opened" << endl;
        return;
    }
    VideoWriter wrt(outputName, VideoWriter::fourcc('X', 'V', 'I', 'D'), cap.get(CV_CAP_PROP_FPS),
        Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)), true);
    if (!wrt.isOpened())
    {
        cerr << "Videowrite failed to open" << endl; return;
    }

    Mat frame;
    namedWindow("video", WINDOW_AUTOSIZE);
    cap.set(CAP_PROP_POS_FRAMES, startFrameIndex);
    int totalFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    clock_t start = clock();

    for (;;)
    {
        cap >> frame; // get a new frame from cap
        if (frame.empty())
            break;
        int nextFrame = cap.get(CV_CAP_PROP_POS_FRAMES);
        clock_t duration = clock() - start;
        //Inpaint with the mask 
        inpaint(frame, mask, frame, 3, INPAINT_TELEA);
        putText(frame, to_string(nextFrame), Point(128, 70),
            FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 2, 8);
        wrt << frame;

        cout << "next frame: " << nextFrame
            << "\nprocessed frames : " << nextFrame - startFrameIndex
            << "\nremaining frame: " << totalFrames - nextFrame
            << "\nTotal Duration :" << duration / CLOCKS_PER_SEC << "s" 
            << "\nProcessing time per frame" 
            << duration / CLOCKS_PER_SEC/ (nextFrame - startFrameIndex) << "s"
            << endl;
        imshow("video", frame);

        char key = (char)waitKey(30);
        switch (key) {
        case 'q':
        case 'Q':
        case 27: //escape key
            return;
        default:
            break;
        }
    }

}

void eventLoop()
{
    for (; ;)
    {
        imshow(wndName, image);
        imshow(wndName2, mask);
        int c = waitKey(0);
        if ((c & 255) == 27)
        {
            cout << "Exiting" << endl;
            break;
        }
        else if (c - '0' < 10 && c - '0' >0)
        {
            circleRadius = c - '0';
        }
        switch (char(c))
        {
        case 's':
            flag = 's'; cout << char(flag) << endl;
            cout << "point out a net line at each corner of the image, 4 in total"
                << "\nFirst, left bottom corner. Second, right bottom corner."
                << "\nThird, left top corner. Fourth, left bottom corner."
                << endl;
            break;
        case 'e':
            if (flag == '0')
            {
                eraseMode = eraseMode ? false : true;
            }
            break;
        case 'i':
            flag = 'i'; cout << char(flag) << endl;
            cout << "Inpainting with mask" << endl;
            inpaintWithMask();
            break;
        case 'f':
            flag = 'f'; cout << char(flag) << endl;
            defineFourEdges(angles, lineLength);
            break;
        case 'm':
            maskDisplaying = maskDisplaying ? false : true;
            maskDisplaying ? maskDisplay() : image0.copyTo(image);
            cout << "Displaying mask at Mask window" << endl;
            break;
        case 'c':
            flag = 'c'; cout << char(flag) << endl;
            findNet(image, angles, lineLength);
            break;
        case 'r':
            flag = 'n'; cout << char(flag) << endl;//back to normal 
            vector<Point>().swap(points);
            vector<double>().swap(angles);
            lineLength = 0;
            image0.copyTo(image);
            mask = Scalar::all(0);//clear the mat
            cout << "Original image is restored\n";
            break;
        case '0':
            flag = '0'; cout << char(flag) << endl;
            cout << "Manual Compensation Mode" << endl;
            break;
        case 'v':
            flag = 'v'; cout << char(flag) << endl;
            cout << "video Mode" << endl;
            processVideo(0);
            break;
        case 'd':
            pressDraw = pressDraw ? false : true;
            break;
        default:
            break;
        }
    }
}

//To define the angle filter by pointing 4 lines. First, "\". 
//Second, "/". Third, "-" but rotating upwards. Finally, "-" downwards
//And make the half length of the longest line as the lineLength filter
void defineFourEdges(vector<double> angles, double lineLength)
{
    double l = lineLength = 0;
    for (size_t i = 0; i < points.size() - 1; i += 2)
    {
        angles.push_back(angle(points[i], points[i + 1]));
        cout << " angles[" << i / 2 << "] " << angles.back() << endl;
        l = length(points[i], points[i + 1]);
        if (lineLength < l)
            lineLength = l;
    }
    lineLength /= 2;
    cout << "length is " << lineLength << endl;

}

//Dilate if the net is too thick, and erode if the net is too thin. 
void findNet(const Mat& image, vector<double> angles, double lineLength)
{
    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
    //pyrUp(pyr, timg, image.size());
    //imshow("pyrDown", pyr);
    cvtColor(image, gray, CV_BGR2GRAY);
    //Dilation(gray, gray);
    Erosion(gray, gray);
    Canny(gray, edges, 20, 80);
    imshow("Edges", edges);

    /// Create Trackbars for Thresholds
    char thresh_label[50];
    sprintf(thresh_label, "Thres: %d + input", min_threshold);
    createTrackbar(thresh_label,wndName, &trackbar, max_trackbar, findLines);
    findLines(0, 0);
    
}

void findLines(int, void*)
{
    vector<Vec4i> lines;
    int minLineLength = 30, maxLineGap = 10;// user change on the interface 

    HoughLinesP(edges, lines, 1, CV_PI / 180, min_threshold + trackbar, minLineLength, maxLineGap);

    makeMask(lines, angles, lineLength);
    image0.copyTo(image);
    inpaint(image, mask, image, 3, INPAINT_TELEA);
}

//Erode the image
void Erosion(Mat& src, Mat& erosion_dst)
{
    int erosion_type;
    int erosion_elem = 0;
    int erosion_size = 0;
    int const max_elem = 2;
    int const max_kernel_size = 21;
    if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
    else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

    Mat element = getStructuringElement(erosion_type,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size));

    /// Apply the erosion operation
    erode(src, erosion_dst, element);
    imshow("Erosion Demo", erosion_dst);
}
//Dilate the image
void Dilation(Mat& src, Mat& dilation_dst)
{
    int dilation_type;
    int dilation_elem = 0;
    int dilation_size = 1;
    int const max_elem = 2;
    int const max_kernel_size = 21;

    if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
    else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

    Mat element = getStructuringElement(dilation_type,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    dilate(src, dilation_dst, element);
    imshow("Dilation Demo", dilation_dst);
}

//Inpaint with Mask
void inpaintWithMask()
{
    image0.copyTo(image);//restore before inpainting 
    inpaint(image, mask, image, 3, INPAINT_TELEA);
}

//Display the mask to the image window
void maskDisplay()
{
    Mat blank(image.size(), image.type(), Scalar::all(255));
    add(image, blank, image, mask, -1);
}

//Generate the mask frome the results of Lines obtained
void makeMask(const vector<Vec4i> lines,
    const vector<double> angles, const double lineLength)
{
    int count = 0;
    const int lineThickness = 3;
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];

        Point p1 = Point(l[0], l[1]), p2 = Point(l[2], l[3]);
        double theta = angle(p1, p2);
        if (((theta >= angles[0] - 2) && (theta <= 90)) || ((theta <= angles[1] + 2) && (theta >= -90)) ||
            (theta >= angles[2] - 1) && (theta <= angles[3] + 1))
        {
            if ((length(p1, p2) < lineLength) /*&& (ifAtCenter(p1, p2, image))*/)
            {
                continue;
            }
            line(mask, p1, p2, Scalar::all(255), lineThickness, 8, 0);
            count++;
        }
    }
    cout << "\nAutogenerated mask's lines.size() = " << lines.size()
        << "\nActually painted lines after angle and length filter = " << count << endl;
}

//the angle between p2->p1 with the horizontal direction
static double angle(Point p1, Point p2)
{
    double dy = p2.y - p1.y;
    double dx = p2.x - p1.x;
    return atan(dy / dx) * 180 / CV_PI;
}
//Calculate the length between two points
static double length(Point p1, Point p2)
{
    double dy = p2.y - p1.y;
    double dx = p2.x - p1.x;
    return sqrt(dx*dx + dy*dy);
}

//Assistive function for specific situation
bool ifAtCenter(Point p1, Point p2, const Mat & image)
{
    Rect centerRect = Rect(175, 113,
        800, 200);
    if (p1.inside(centerRect) && p2.inside(centerRect))
    {
        return true;
    }
    else return false;
}

static void onMouse(int event, int x, int y, int, void*)
{
    if (event == EVENT_LBUTTONDOWN)
        mouseDown = true;
    else if (event == EVENT_LBUTTONUP)
        mouseDown = false;

    Point seed = Point(x, y);

    switch (flag)
    {
    case 's':
        if (event == EVENT_LBUTTONDOWN)
        {
            points.push_back(seed);
            circle(image, points.back(), 2, Scalar::all(255), CV_FILLED, 8, 0);
            circle(mask, points.back(), 2, Scalar::all(255), CV_FILLED, 8, 0);
            if (points.size() % 2 == 0)
            {
                line(image, points.back(), *(&points.back() - 1), Scalar::all(255), 1, 8, 0);
                line(mask, points.back(), *(&points.back() - 1), Scalar::all(255), 1, 8, 0);
            }
            if (points.size() > 7)
            {
                cout << "selection complete" << endl;
                flag = 'n'; cout << char(flag);
            }
            cout << seed << endl;
            cout << "**************************" << endl;
            imshow(wndName, image);
            imshow(wndName2, mask);
            break;
        }

    case '0':
        if (mouseDown)
        {
            if (event == EVENT_MOUSEMOVE)
            {
                Scalar s = eraseMode ? Scalar::all(0) : Scalar::all(255);
                circle(image, seed, circleRadius, s, CV_FILLED);
                circle(mask, seed, circleRadius, s, CV_FILLED);
            }
        }
        imshow(wndName, image);
        imshow(wndName2, mask);
        break;
    default:
        break;
    }


}
static void onMouseForMaskWnd(int event, int x, int y, int, void*)
{
    if (event == EVENT_LBUTTONDOWN)
        mouseDown = true;
    else if (event == EVENT_LBUTTONUP)
        mouseDown = false;
    //Eraser on mask
    if (mouseDown)
    {
        if (event == EVENT_MOUSEMOVE)
        {
            circle(mask, Point(x, y), circleRadius, Scalar::all(0), CV_FILLED);
            imshow(wndName2, mask);
        }
    }
}

//HELP info
void help()
{
    cout << "Basic steps:\n"
        "1. 's' to select the four lines \n"
        "2. 'f' to define the four edges \n"
        "3. 'c' to autogenerate mask and inpaint"
        "4a. '0' to manually draw mask on the first result\n"
        "4b. Right button down and drag on the mask to erase wrongly generated part\n"
        "5. 'i' to inpaint with the mask\n"
        "6. repeat 4 and 5 until satisfying mask is obtained\n"
        "7. 'v' to inpaint the video and output result video\n\n"
        << endl;
    cout << "Hot keys: \n"
        "\tESC - quit the program or the video mode\n"
        "\ts - selection the four edges\n"
        "\tc - comfirm and start inpainting\n"
        "\tm - switch mask mode\n"
        "\tr - restore the original image\n"
        "\tf - define the four edges\n"
        "\ti - inpaint with the mask\n"
        "\t0 - manual compensation\n"
        "\tNumber keys can control the radius of the eraser on mask\n"
        << endl;
}