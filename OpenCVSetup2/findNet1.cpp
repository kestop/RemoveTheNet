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
const string videoName("./videos/ActionIndoorSportsBristolCam2.mp4");
const string outputName("./output/sample4.avi");
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
void generateTheNet();
int getGridPointsNumber(double l1, double l2, double ld);
void predictGapJoints(vector<vector<Point>> &pp);
void initJtss(const vector<vector<Point>>& pp, vector<vector<Point>>& Jtss);
void extendTopOrLeft(vector<vector<Point>>& pp, vector<vector<Point>>& Jtss, double dx, double dy, size_t i, size_t sz);
void expandJointsToWholeNet(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss);
bool outOfImage(Point p);
double getTheD(vector<Point> p, int xORy);
void predictPoints(const vector<cv::Point> pts, vector<double>& gapJts);
void avgGapJoints(const vector<vector<double>> &gJ, double fGJ[]);
void generateJoints(double fGJ[], vector<Point> &pts);
bool nextLine(Point seed);




vector<Point> points;
vector<vector<Point>> priorPoints;
vector<Point> priorRow;
vector<vector<Point>> horJtss;
vector<vector<Point>> verJtss;//hardcode 100

int flag = 'n';
int min_threshold = 50;
int max_trackbar = 150;
int trackbar = max_trackbar / 2;
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
    capAFrameFromVideo(51);

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
//@ frameIndex The location of the frame
void capAFrameFromVideo(const int frameIndex)
{
    VideoCapture cap(videoName);
    if (!cap.isOpened())
    {
        cerr << "Video cannot be opened" << endl;
        exit(EXIT_FAILURE);
    }
    cap.set(CAP_PROP_POS_FRAMES, frameIndex);
    cap >> image0;
    cout << "Total frame number is " << cap.get(CV_CAP_PROP_FRAME_COUNT) << "\n"
        << "Start frame index is " << cap.get(CAP_PROP_POS_FRAMES) << "\n"
        << "FPS is " << cap.get(CAP_PROP_FPS)
        << endl;
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
            << duration / CLOCKS_PER_SEC / (nextFrame - startFrameIndex) << "s"
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
            //for new features
        case 'l':
            flag = 'l'; cout << char(flag) << endl;
            cout << "Prior Points selection" << endl;
            break;
        case 'a':
            flag = 'a'; cout << char(flag) << endl;
            cout << "Predict points and draw them" << endl;
            generateTheNet();
            break;
        default:
            break;
        }
    }
}
//@brief Count the distance using the edges and the prior points
//invoking predictPoints()
void generateTheNet()
{
    priorPoints.push_back(priorRow);
    //vertical 
    vector<vector<Point>> verticalPtss;
    for (size_t i = 0; i < priorPoints[0].size(); i++)//priorPoints's cols have the same size 
    {
        vector<Point> verticalPts;
        for (size_t j = 0; j < priorPoints.size(); j++)
        {
            verticalPts.push_back(priorPoints[j][i]);
        }
        verticalPtss.push_back(verticalPts);
    }

    predictGapJoints(verticalPtss);
    expandJointsToWholeNet(verticalPtss,horJtss);

    polylines(image, horJtss, false, Scalar(0, 255, 0), 1);
    
    //make horizontal Ptss
    //vector < vector<Point> > horizontalPtss;
    //for (size_t i = 0; i < verticalPtss[0].size(); i++)// what if Pts.size() is larger at the end?
    //{
    //    vector<Point> horizontalPts;
    //    for (size_t j = 0; j < verticalPtss.size(); j++)
    //    {
    //        horizontalPts.push_back(verticalPtss[j][i]);
    //    }
    //    horizontalPtss.push_back(horizontalPts);
    //}

    //predictGapJoints(horizontalPtss);
    //expandJointsToWholeNet(horizontalPtss);
    //final vertical 
    //vector<vector<Point>> verticalPtss2;
    //for (size_t i = 0; i < horizontalPtss[0].size(); i++)
    //{
    //    vector<Point> verticalPts2;
    //    for (size_t j = 0; j < horizontalPtss.size(); j++)
    //    {
    //        verticalPts2.push_back(horizontalPtss[j][i]);
    //    }
    //    verticalPtss2.push_back(verticalPts2);
    //}

    //polylines(image, verticalPtss2, false, Scalar::all(255), 2);
    //polylines(image, horizontalPtss, false, Scalar::all(255), 1);
    ////polylines(mask, verticalPtss2, false, Scalar::all(255), 2);
    //polylines(mask, horizontalPtss, false, Scalar::all(255), 1);



    imshow(wndName, image);
    imshow(wndName2, mask);
}

void predictGapJoints(vector<vector<Point>> &pp)
{
    int n = 0;
    const size_t sz = pp.size();
    vector<vector<double>> gapJointsNum(sz);

    double finalGapJointsNum[10] = { 0 };

    cout << "Gap Size : " << gapJointsNum.size() << endl;
    for (size_t i = 0; i < sz; i++)
    {
        predictPoints(pp[i], gapJointsNum[i]);
    }
    avgGapJoints(gapJointsNum, finalGapJointsNum);
    for (size_t i = 0; i < sz; i++)
    {
        generateJoints(finalGapJointsNum, pp[i]);
    }
}
//put the square into the Jtss
void initJtss(const vector<vector<Point>> &pp, vector<vector<Point>> &Jtss)
{
    //Jtss.size() == pp[0].size();
    for (size_t i = 0; i < pp[0].size(); i++)
    {
        Jtss.push_back(vector<Point>());
    }
    for (size_t i = 0; i < pp[0].size(); i++)
    {
        for (size_t j = 0; j < pp.size(); j++)
        {
            Jtss[i].push_back(pp[j][i]);
        }
    }
}

void extendTopOrLeft(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss, double dx, double dy, size_t i, size_t sz)
{
    vector<Point> vecTmp;
    for (size_t j = 1; ; j++)
    {
        double pX = pp[i].front().x + dx*j;
        double pY = pp[i].front().y + dy*j;
        Point tmp = Point(pX, pY);
        if (!outOfImage(tmp))
            break;
        vecTmp.push_back(tmp);
        if (j > Jtss.size() - sz)
        {
            Jtss.push_back(vector<Point>());
        }
        Jtss[j + sz - 1].push_back(tmp);

        circle(image, tmp, 3, Scalar(0, 0, 255), -1);
        circle(mask, tmp, 3, Scalar(255), -1);
    }
    pp[i].insert(pp[i].begin(), vecTmp.rbegin(), vecTmp.rend());


    vecTmp.clear();
}

void extendBottomOrRight(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss, double dx, double dy, size_t i, size_t sz)
{
    for (size_t k = 1;; k++)
    {
        cout << "/n *pp[i].end() : " << (*(&(pp[i].back()) - k + 1)).y << endl;
        double pX = (*(&(pp[i].back()) - k + 1)).x - dx*k;
        double pY = (*(&(pp[i].back()) - k + 1)).y - dy*k;
        cout << "\npX " << pX << endl;
        cout << "\npY " << pY << endl;
        Point tmp = Point(int(round(pX)), int(round(pY)));
        cout << "\ntmp in bottom gene : " << tmp << endl;
        if (!outOfImage(tmp))
            break;
        pp[i].push_back(tmp);
        circle(image, tmp, 3, Scalar(0, 0, 255), -1);
        circle(mask, tmp, 3, Scalar(255), -1);
    }
}

void expandJointsToWholeNet(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss)//???
{
    initJtss(pp, Jtss);
    const size_t sz = pp[0].size();
    //go through joints vector
    for (size_t i = 0; i < pp.size(); i++)
    {
        double dX = getTheD(pp[i], 1);
        double dY = getTheD(pp[i], 0);
        //top or left part

        extendTopOrLeft(pp, Jtss, dX, dY, i, sz);

        //the bottom or right part
        extendBottomOrRight(pp, Jtss, dX, dY, i, sz);
    }
}

bool outOfImage(Point p)
{
    if (p.inside(Rect(0, 0, image.cols, image.rows)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

/*@brief To get the incremental distance in x or y direction
@param p
@param xORy >0 in x, else y
*/
double getTheD(vector<Point> p, int xORy)
{
    double sum = 0;
    const size_t sz = p.size();
    if (xORy > 0)
    {
        for (size_t i = 0; i < sz - 1; i++)
        {
            sum += (p[i].x - p[i + 1].x);
        }
    }
    else
    {
        for (size_t i = 0; i < sz - 1; i++)
        {
            sum += (p[i].y - p[i + 1].y);
        }
    }
    return (sum /= sz);
}


/** @brief To Predict Points on a prior knowledge of user defined points
The points has to be started with one grid segement, not two or more.
Work on both vertical and horizontal situations.
@param pts A vector of cv::Point to detemine a line. Number of points needs to be even.
*/
void predictPoints(const vector<cv::Point> pts, vector<double> &gapJtsNum)
{
    const size_t sz = pts.size();
    for (size_t i = 0; i < sz - 3; i++)
    {
        double l1 = length(pts[i], pts[i + 1]);
        double l2 = length(pts[i + 1], pts[i + 2]);
        //To generate or not
        if (l2 > 2 * l1)
        {
            double ld = l2;
            l2 = length(pts[i + 2], pts[i + 3]);
            int n = getGridPointsNumber(l1, l2, ld);
            gapJtsNum.push_back(n);
        }
    }
}

/*@brief To calculate the avg joints number in gaps
@param gj vector<vector<double>> matrix stored the number of joints in gaps
@param [] to store the average number of joints for user defined joints
*/
void avgGapJoints(const vector<vector<double>> &gJ, double fGJ[])
{
    for (size_t col = 0; col < gJ.front().size(); col++)
    {
        for (size_t row = 0; row < gJ.size(); row++)
        {
            fGJ[col] += gJ[row][col];
        }
        fGJ[col] /= (int)gJ.size();
        cout << "/nfGJ[" << col << "] = " << fGJ[col] << endl;
    }
}

/*@brief To generate the joints between user defined joints
@param double [] the number of joints in gaps
@param pts Handled points vector
*/
void generateJoints(double fGJ[], vector<Point> &pts)
{
    int k = 0;
    for (size_t i = 0; i < pts.size() - 3; i++)
    {
        int n = int(round(fGJ[k]));
        double l1 = length(pts[i], pts[i + 1]);
        double l2 = length(pts[i + 1], pts[i + 2]);
        if (l2 > 2 * l1)
        {
            l2 = length(pts[i + 2], pts[i + 3]);
            double dX = ((pts[i].x - pts[i + 2].x) +
                (pts[i + 1].x - pts[i + 3].x)) / (2.0 * (n + 1));
            double dY = ((pts[i].y - pts[i + 2].y) +
                (pts[i + 1].y - pts[i + 3].y)) / (2.0 * (n + 1));
            cout << "DX : " << dX << endl;
            cout << "DY : " << dY << endl;

            vector<Point> vecTmp;
            for (size_t j = 1; j < n; j++)
            {
                Point tmp = Point(round(pts[i + 1].x - j*dX), round(pts[i + 1].y - j*dY));
                cout << tmp << endl;
                vecTmp.push_back(tmp);
                circle(image, tmp, 2, Scalar(0, 255, 0), FILLED);
                circle(mask, tmp, 2, Scalar::all(255), FILLED);
            }
            //Insert to keep ordered
            pts.insert(pts.begin() + i + 2, vecTmp.begin(), vecTmp.end());
            i += vecTmp.size();
            k++;
        }
    }

}


/** @brief To get the number of the line segements, when l1 > l2, we have :
ds = (l1-l2)/(n+1);
(l2 + ds)+(l2 + 2*ds)+....+(l2+n*ds) = ld => n*l2 + (n(n+1)/2)*ds = ld => n = 2c/(a+b)
@param l1 line segement 1
@param l2 line segement 2
@param ld distance between 2 line segements
*/
int getGridPointsNumber(double l1, double l2, double ld)
{
    double n = (2 * ld) / (l1 + l2);
    cout << "n: " << n << endl;
    return int(round(n));
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
    createTrackbar(thresh_label, wndName, &trackbar, max_trackbar, findLines);
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
        }
        break;

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
    case 'l':
        if (event == EVENT_LBUTTONDOWN)
        {
            if (priorRow.size() > 1 && nextLine(seed))
            {
                priorPoints.push_back(priorRow);
                priorRow.clear();
            }
            priorRow.push_back(seed);
            circle(image, seed, 2, Scalar::all(255), CV_FILLED, 8, 0);
            circle(mask, seed, 2, Scalar::all(255), CV_FILLED, 8, 0);
            cout << seed << endl;
            cout << "**************************" << endl;
            imshow(wndName, image);
            imshow(wndName2, mask);
        }
        break;
    default:
        break;
    }


}

bool nextLine(Point seed)
{
    return seed.x < priorRow[1].x ? true : false;
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