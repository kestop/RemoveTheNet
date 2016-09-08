#include <opencv2\highgui.hpp>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\photo.hpp>
#include <opencv2\videoio.hpp>


#include "Header.h"

vector<Point> points;
vector<Point> priorRow;
vector<vector<Point>> priorPoints;
vector<vector<Point>> horJtss;
vector<vector<Point>> verJtss;

int flag = 'n';
int min_threshold = 50;
int max_trackbar = 150;
int trackbar = max_trackbar / 2;
int circleRadius = 2;
bool DilateOrErode = true;
bool pyrDownOrUp = false;
bool eraseMode = false;
bool mouseDown = false;
bool maskDisplaying = false;
bool pressDraw = true;//not applied yet
vector<double> angles = { 81.9,-79.8,-1.22,2.87 }; //default angels and lineLength \ / -up -down 81.9 -79.8 -1.22 2.87
double lineLength = 130;

#define DEBUG

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
            processVideo(51);
            break;
        case 'd':
            pressDraw = pressDraw ? false : true;
            break;
            //for new features
            //'l' allows to add more rows but cannot modify current row length 
        case 'l':
            flag = 'l'; cout << char(flag) << endl;

            //hardPriorPts();
            verJtss.clear();
            horJtss.clear();
            image0.copyTo(image);
            mask = Scalar::all(0);

            for each (auto p in priorPoints)
            {
                for each (auto var in p)
                {
                    circle(image, var, 2, Scalar::all(255), CV_FILLED, 8, 0);
                    circle(mask, var, 2, Scalar::all(255), CV_FILLED, 8, 0);
                }
            }
            cout << "Prior Points Selection. Press 'l' again to restore image and modify base points." << endl;
            break;
        case 'a':
            flag = 'a'; cout << char(flag) << endl;
            cout << "Predict points and draw them" << endl;
            generateTheNet();
            break;
        case 'b':
            flag = 'b'; cout << char(flag) << endl;
            drawTheNet();
            break;

            //clear everything and restart
        case 'z':
            flag = 'l'; cout << char(flag) << endl;
            image0.copyTo(image);
            mask = Scalar::all(0);
            verJtss.clear();
            horJtss.clear();
            priorPoints.clear();
            priorRow.clear();
            break;

        case 'u':
            flag = 'u';

            break;
        case 'y':
            flag = 'y';
            templateMatching();
            break;
        default:
            break;
        }
    }
}


//templateMatching
void templateMatching()
{
    int result_cols = image.cols - sample.cols + 1;
    int result_rows = image.rows - sample.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);
    matchTemplate(image, sample, result, CV_TM_CCOEFF);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;

    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better

    matchLoc = maxLoc;

    /// Show me what you got
    rectangle(image, matchLoc, Point(matchLoc.x + sample.cols, matchLoc.y + sample.rows), Scalar::all(0), 2, 8, 0);
    rectangle(result, matchLoc, Point(matchLoc.x + sample.cols, matchLoc.y + sample.rows), Scalar::all(0), 2, 8, 0);

}

//sort vector to ascending order(top to bottom or left to right). 1 in y , 0 in x
void sortVector(vector<vector<Point>> &pp, int xORy)
{
    for (size_t i = 0; i < pp.size(); i++)
    {
        for (size_t j = 0; j < pp[i].size() - 1; j++)
        {
            for (size_t k = 0; k < pp[i].size() - j - 1; k++)
            {
                if (xORy > 0)
                {
                    if (pp[i][k].y > pp[i][k + 1].y)
                    {
                        Point tmp = pp[i][k + 1];
                        pp[i][k + 1] = pp[i][k];
                        pp[i][k] = tmp;
                    }
                }
                else
                {
                    if (pp[i][k].x > pp[i][k + 1].x)
                    {
                        Point tmp = pp[i][k + 1];
                        pp[i][k + 1] = pp[i][k];
                        pp[i][k] = tmp;
                    }
                }
            }
        }
    }
}
//unfinished
void quadraticFit(vector<Point> &p)
{
    const size_t N = p.size();
    const int n = 2;

    double X[2 * n + 1];
    size_t i, j, k;
    for (i = 0; i < 2 * n + 1; i++)
    {
        X[i] = 0;
        for (j = 0; j < N; j++)
        {
            X[i] = X[i] + pow(p[j].x, i);
        }
    }
    double B[n + 1][n + 2], a[n + 1];

    for (i = 0; i <= n; i++)
        for (j = 0; j <= n; j++)
            B[i][j] = X[i + j];            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
    double Y[n + 1];                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    for (i = 0; i < n + 1; i++)
    {
        Y[i] = 0;
        for (j = 0; j < N; j++)
            Y[i] = Y[i] + pow(p[j].x, i)*p[j].y;        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    }
    for (i = 0; i <= n; i++)
    {
        B[i][n + 1] = Y[i];
    }//load the values of Y as the last column of B(Normal Matrix but augmented)
    int n1 = 3;             //n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations
    cout << "\nThe Normal(Augmented Matrix) is as follows:\n";
    for (i = 0; i < n1; i++)            //print the Normal-augmented matrix
    {
        for (j = 0; j <= n1; j++)
            cout << B[i][j] << setw(16);
        cout << "\n";
    }
    for (i = 0; i < n1; i++)                    //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
        for (k = i + 1; k < n1; k++)
            if (B[i][i] < B[k][i])
                for (j = 0; j <= n1; j++)
                {
                    double temp = B[i][j];
                    B[i][j] = B[k][j];
                    B[k][j] = temp;
                }

    for (i = 0; i < n1 - 1; i++)            //loop to perform the gauss elimination
        for (k = i + 1; k < n1; k++)
        {
            double t = B[k][i] / B[i][i];
            for (j = 0; j <= n1; j++)
                B[k][j] = B[k][j] - t*B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
        }
    for (i = n1 - 1; i >= 0; i--)                //back-substitution
    {                        //x is an array whose values correspond to the values of x,y,z..
        a[i] = B[i][n1];                //make the variable to be calculated equal to the rhs of the last equation
        for (j = 0; j < n1; j++)
            if (j != i)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
                a[i] = a[i] - B[i][j] * a[j];
        a[i] = a[i] / B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
    }
    cout << "\nThe values of the coefficients are as follows:\n";
    for (i = 0; i < n1; i++)
        cout << "x^" << i << "=" << a[i] << endl;            // Print the values of x^0,x^1,x^2,x^3,....    
    cout << "\nHence the fitted Polynomial is given by:\ny=";
    for (i = 0; i < n1; i++)
        cout << " + (" << a[i] << ")" << "x^" << i;
    cout << "\n";

}
void linearFit(const vector<Point>& p, double & a, double & b)
{
    int i, j, k, n;
    cout << "\nEnter the no. of data pairs to be entered:\n";        //To find the size of arrays
    n = p.size();
    vector<double> x(n);
    vector<double> y(n);
    //Input x-values
    for (i = 0; i < n; i++)
        x[i] = p[i].x;
    //Input y-values
    for (i = 0; i < n; i++)
        y[i] = p[i].y;

    double xsum = 0, x2sum = 0, ysum = 0, xysum = 0;                //variables for sums/sigma of xi,yi,xi^2,xiyi etc
    for (i = 0; i < n; i++)
    {
        xsum = xsum + x[i];                        //calculate sigma(xi)
        ysum = ysum + y[i];                        //calculate sigma(yi)
        x2sum = x2sum + pow(x[i], 2);                //calculate sigma(x^2i)
        xysum = xysum + x[i] * y[i];                    //calculate sigma(xi*yi)
    }
    a = (n*xysum - xsum*ysum) / (n*x2sum - xsum*xsum);            //calculate slope
    b = (x2sum*ysum - xsum*xysum) / (x2sum*n - xsum*xsum);            //calculate intercept

                                                                      //double y_fit[n];                        //an array to store the new fitted values of y    
    vector<double> y_fit(n);
    for (i = 0; i < n; i++)
        y_fit[i] = a*x[i] + b;                    //to calculate y(fitted) at given x points
    cout << "S.no" << setw(5) << "x" << setw(19) << "y(observed)" << setw(19) << "y(fitted)" << endl;
    cout << "-----------------------------------------------------------------\n";
    for (i = 0; i < n; i++)
        cout << i + 1 << "." << setw(8) << x[i] << setw(15) << y[i] << setw(18) << y_fit[i] << endl;//print a table of x,y(obs.) and y(fit.)    
    cout << "\nThe linear fit line is of the form:\n\n" << a << "x + " << b << endl;        //print the best fit line
}
/*@brief Top level fucntion when using user-define points technique.
Based on priorPoints[][]
horJtss's order is weird. Top part(from up to bottom), 18 17 16... 12; mid part, 0....11; bottom part 19 .. 24.
which will project to the structure of verJtss. Making verJtss start at the mid up to mid bottom then to mid top to top top ,
then from bottom top to bottom bottom.
BUT horJtss's each row are alright, all start from left to right ()
*/
void generateTheNet()
{
    if (!priorRow.empty())
    {
        priorPoints.push_back(priorRow);
        priorRow.clear();
    }

    //change to vertical . Actually, we can skip this 
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
    //quadraticFit(verticalPtss[0]);
    //linearFit(verticalPtss[0]);

    predictGapJoints(verticalPtss);
    expandJointsToWholeNet(verticalPtss, horJtss);

    predictGapJoints(horJtss);

    //initJtss(horJtss, verJtss);
    expandJointsToWholeNet(horJtss, verJtss);
    sortVector(verJtss, 1);
    tideUpJtss(verJtss);
    return;
}

void tideUpJtss(vector<vector<Point>> &pp)
{
    for (size_t i = 0; i < pp.size(); i++)
    {
        if (length(pp[i].back(), (*(pp[i].end() - 2))) >
            1.3 * length((*(pp[i].end() - 2)), (*(pp[i].end() - 3)))
            )

        {
            pp[i].pop_back();

        }
    }

    //image0.copyTo(image);
    //mask = Scalar::all(0);

    //for each (auto p in verJtss)
    //{
    //    for each (auto var in p)
    //    {
    //        circle(image, var, 2, Scalar::all(255), CV_FILLED, 8, 0);
    //        circle(mask, var, 2, Scalar::all(255), CV_FILLED, 8, 0);
    //    }
    //}

    //for each (auto p in horJtss)
    //{
    //    for each (auto var in p)
    //    {
    //        circle(image, var, 2, Scalar::all(255), CV_FILLED, 8, 0);
    //        circle(mask, var, 2, Scalar::all(255), CV_FILLED, 8, 0);
    //    }
    //}

}


//put the square into the Jtss. Order up to down : 0 - n
void initJtss(const vector<vector<Point>> &pp, vector<vector<Point>> &Jtss)
{
    //Jtss.size() == pp[0].size();
    for (size_t i = 0; i < pp[pp.size() / 2].size(); i++)
    {
        Jtss.push_back(vector<Point>());
    }
    size_t j = 0;
    for (size_t i = 0; i < pp[pp.size() / 2].size(); i++)
    {
        for (j = 0; j < pp.size(); j++)
        {
            if (i < pp[j].size())
            {
                Jtss[i].push_back(pp[j][i]);
            }
            else
            {
                cout << "yichu loacation j :" << j << "i" << i << endl;
            }

        }
    }
}

void expandJointsToWholeNet(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss)//???
{
    initJtss(pp, Jtss);
    const size_t sz = pp[0].size();
    //top or left part
    double dx[100];
    double dy[100];
    for (size_t i = 0; i < pp.size(); i++)
    {
        double dX = getTheD(pp[i], 1);
        dx[i] = dX;
        double dY = getTheD(pp[i], 0);
        dy[i] = dY;
        extendTopOrLeft(pp, Jtss, dx[i], dy[i], i, sz);
    }
    //the bottom or right part
    const size_t sz2 = Jtss.size();
    for (size_t i = 0; i < pp.size(); i++)
    {
        extendBottomOrRight(pp, Jtss, dx[i], dy[i], i, sz2);
    }
}

void extendTopOrLeft(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss,
    double dx, double dy, size_t i, size_t sz)
{
    vector<Point> vecTmp;
    for (size_t j = 1; ; j++)
    {
        //double a,b;
        //linearFit(pp[i], a, b);
        //double pX = round(pp[i].front().x + dx*j);
        //double pY = a*pX + b;



        double pX = round(pp[i].front().x + dx*j);
        double pY = round(pp[i].front().y + dy*j);
        Point tmp = Point(int(round(pX)), int(round(pY)));
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

void extendBottomOrRight(vector<vector<Point>> &pp, vector<vector<Point>> &Jtss,
    double dx, double dy, size_t i, size_t sz)
{

    for (size_t k = 1;; k++)
    {
        double pX = (*(&(pp[i].back()) - k + 1)).x - dx*k;
        double pY = (*(&(pp[i].back()) - k + 1)).y - dy*k;

        Point tmp = Point(int(round(pX)), int(round(pY)));
        if (!outOfImage(tmp))
            break;
        pp[i].push_back(tmp);
        if (k > Jtss.size() - sz)
        {
            Jtss.push_back(vector<Point>());
        }
        //if ((verJtss.size() >= 17) &&(verJtss[17].size() >= 22))
        //{
        //    cout << "mingzhong" << endl;
        //    cout << "pX = " << pX << "py = " << 
        //        pY <<"dx " <<dx << "dy " <<dy  
        //        << "last point "<<pp[i].back() 
        //        << endl;
        //}
        Jtss[k + sz - 1].push_back(tmp);

        circle(image, tmp, 3, Scalar(0, 0, 255), -1);
        circle(mask, tmp, 3, Scalar(255), -1);
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
/*@brief To predict the joints within the square
*/
void predictGapJoints(vector<vector<Point>> &pp)
{
    int n = 0;
    const size_t sz = pp.size();
    vector<vector<double>> gapJointsNum(sz);

    double finalGapJointsNum[10] = { 0 };

    cout << "Gap Size : " << gapJointsNum.size() << endl;
    for (size_t i = 0; i < sz; i++)
    {
        predictGapPointsNum(pp[i], gapJointsNum[i]);
    }
    cout << "enter avgGap " << endl;
    avgGapJoints(gapJointsNum, finalGapJointsNum);
    for (size_t i = 0; i < sz; i++)
    {
        generateGapJoints(finalGapJointsNum, pp[i]);
    }
}

/** @brief To Predict Points on a prior knowledge of user defined points
The points has to be started with one grid segement, not two or more.
Work on both vertical and horizontal situations.
@param pts A vector of cv::Point to detemine a line. Number of points needs to be even.
*/
void predictGapPointsNum(const vector<cv::Point> pts, vector<double> &gapJtsNum)
{
    const size_t sz = pts.size();
    if (pts.size() < 4)
    {
        gapJtsNum.push_back(0);
    }
    else
    {
        for (size_t i = 0; i < sz - 3; i++)
        {

            double l1 = length(pts[i], pts[i + 1]);
            double l2 = length(pts[i + 1], pts[i + 2]);
            if (l2 > 1.5 * l1)
            {
                double ld = l2;
                l2 = length(pts[i + 2], pts[i + 3]);
                int n = getGridPointsNumber(l1, l2, ld);
                cout << "n: " << n << endl;
                gapJtsNum.push_back(n);
            }
        }
    }
}

/*@brief To calculate the avg joints number in gaps
@param gj vector<vector<double>> matrix stored the number of joints in gaps
@param [] to store the average number of joints for user defined joints
*/
void avgGapJoints(const vector<vector<double>> &gapJointsNum, double fGJ[])
{
    for (size_t col = 0; col < gapJointsNum.front().size(); col++)
    {
        for (size_t row = 0; row < gapJointsNum.size(); row++)
        {
            fGJ[col] += gapJointsNum[row][col];
        }
        fGJ[col] /= (int)gapJointsNum.size();
        cout << "/nfGJ[" << col << "] = " << fGJ[col] << endl;
    }
}

/*@brief To generate the joints between user defined joints
@param double [] the number of joints in gaps
@param pts Handled points vector
*/
void generateGapJoints(double fGJ[], vector<Point> &pts)
{
    int k = 0;
    if (pts.size() >= 4)
    {
        for (size_t i = 0; i < pts.size() - 3; i++)
        {
            int n = int(round(fGJ[k]));//n only represents the number of line segements in gap
            double l1 = length(pts[i], pts[i + 1]);
            double l2 = length(pts[i + 1], pts[i + 2]);
            if (l2 > 1.5 * l1)
            {
                l2 = length(pts[i + 2], pts[i + 3]);
                double dX = ((pts[i].x - pts[i + 2].x) +
                    (pts[i + 1].x - pts[i + 3].x)) / (2.0 * (n + 1));
                double dY = ((pts[i].y - pts[i + 2].y) +
                    (pts[i + 1].y - pts[i + 3].y)) / (2.0 * (n + 1));

                vector<Point> vecTmp;
                //cout << "tmp in Generate Gap Points" << endl;
                for (size_t j = 1; j < n; j++)
                {
                    Point tmp = Point(int(round(pts[i + 1].x - j*dX)),
                        int(round(pts[i + 1].y - j*dY)));
                    //cout << tmp << endl;
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
    return int(round(n));
}
/*@brief diff between the c1 and c2 and c3
c1 + n*dd +(n+1)*n/2*ds = c2
c1 + (n+1)*dd + (n+2)*(n+1)*ds/2 = c3

*/
void getDiff(int c0, int c1, int c2, int c3, int n, double &d, double &ds)
{
    double d1, d2, ds1, ds2;
    d1 = (double(c2) - double(n)*c3 / (n + 2) - double(2)*c1 / (n + 2))*((n + 2) / double(n));
    d2 = (double(c2) - double(n + 1)*c3 / (n + 3) - double(2)*c0 / (n + 3))*((n + 3) / double(n + 1));

    ds1 = (double(2) / (n + 1))*(double(c1) / n + c3 - (n + 1)*c2 / double(n));
    ds2 = (double(2) / (n + 2))*(double(c0) / (n + 1) + c3 - (n + 2)*c2 / double(n + 1));

    d = (d1 + d2) / 2;
    ds = (ds1 + ds2) / 2;
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
    Mat tmp;
    //pyrDownOrUp ? pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2)) : pyrUp(pyr, tmg, image.size());
    cvtColor(image, gray, CV_BGR2GRAY);
    // dilate or erode to reduce noise
    DilateOrErode ? Dilation(gray, gray) : Erosion(gray, gray);
    Canny(gray, edges, 25, 50);
    imshow("Edges", edges);

    // Make findLines() real-time adjustbale
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
    int erosion_size = 3;
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
    int dilation_type = 0;
    int dilation_elem = 0;
    int dilation_size = 1;
    //for track bars in future
    /*int const max_elem = 2;
    int const max_kernel_size = 21;*/

    if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
    else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

    Mat element = getStructuringElement(dilation_type,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    dilate(src, dilation_dst, element);
    imshow("Dilation", dilation_dst);
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
    const int lineThickness = 5;
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];

        Point p1 = Point(l[0], l[1]), p2 = Point(l[2], l[3]);
        double theta = angle(p1, p2);
        if (((theta >= angles[0] - 2) && (theta <= 90)) || ((theta <= angles[1] + 2) && (theta >= -90)) ||
            (theta >= angles[2] - 1) && (theta <= angles[3] + 1))
        {
            if ((length(p1, p2) > lineLength) & (ifAtCenter(p1, p2, image)))
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
    case 'u':
        if (event == EVENT_LBUTTONDOWN)
        {
            Rect roi = Rect(seed.x, seed.y, 15, 15);
            cout << roi << endl;
            rectangle(image, roi, Scalar(0, 0, 0), 1);
            sample = image(roi);
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
//helper function
void hardPriorPts()
{
    vector<vector<Point>> pp;
    vector<Point> a;
    a.push_back(Point(338, 190));
    a.push_back(Point(373, 189));
    a.push_back(Point(821, 187));
    a.push_back(Point(859, 187));
    vector<Point> b = { Point(345,222),Point(375,224),Point(821,223),Point(854,226) };
    vector<Point> c = { Point(365, 510), Point(399, 509), Point(810, 513), Point(842, 513) };
    vector<Point> d = { Point(371, 538), Point(400, 539), Point(809, 543), Point(840, 544) };
    vector<Point> e = { Point(395, 481), Point(428, 480), Point(681, 480), Point(711, 481) };
    vector<Point> f = { Point(396, 510), Point(429, 517), Point(673, 510), Point(715, 509) };
    vector<Point> g = { Point(410, 600), Point(436, 600), Point(681, 599), Point(708, 601) };
    vector<Point> h = { Point(407, 626), Point(437, 626), Point(682, 629), Point(715, 630) };
    priorPoints = pp = { a,b,c,d };
}


void drawTheNet()
{
    polylines(image, horJtss, false, Scalar(0, 255, 0), 5);
    polylines(mask, horJtss, false, Scalar::all(255), 5);
    polylines(image, verJtss, false, Scalar(255, 0, 0), 5);
    polylines(mask, verJtss, false, Scalar::all(255), 5);
}


//HELP info
void help()
{
    cout << "Basic steps:\n"
        "1. 's' to select the four lines \n"
        "2. 'f' to define the four edges \n"
        "3. 'c' to autogenerate mask and inpaint"
        "4a. '0' to manually draw mask on the first result\n"
        "4b. '0' again to switch to erase mode; Left button down and drag on the mask to erase wrongly generated part\n"
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