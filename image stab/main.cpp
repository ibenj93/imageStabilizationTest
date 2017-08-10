//
//  main.cpp
//  image stab
//
//  Created by Ibrahim Benjelloun on 10/08/2017.
//  Copyright © 2017 Ibrahim Benjelloun. All rights reserved.
//

#include "opencv2/opencv.hpp"
#include "VideoStabilizer.hpp"

using namespace cv;
using namespace VK;
int main(int argc, char** argv)
{
    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(1))
        return 0;
    for(;;)
    {
        Mat frame;
        cap >> frame;
        
        VideoStabilizer stabilizer;
        
        if(stabilizer.iteration(frame))
           {
               cv::imshow("image stabilisée", stabilizer.getStabilizedOutput()  );
               cv::namedWindow("image stabilisée",WINDOW_NORMAL);
               cv::resizeWindow("image stabilisée", 600,600);
               
               printf("%.4f %.4f %.4f | %.4f %.4f %.4f | %.4f %.4f %.4f  \n",
                      stabilizer.getCurrentTransform().dx,
                      stabilizer.getCurrentTransform().dy,
                      stabilizer.getCurrentTransform().da,
                      stabilizer.getTrajectory().x,
                      stabilizer.getTrajectory().y,
                      stabilizer.getTrajectory().a,
                      stabilizer.getSmoothedTrajectory().x,
                      stabilizer.getSmoothedTrajectory().y,
                      stabilizer.getSmoothedTrajectory().a
                      );
           }
        
        if( frame.empty() ) break; // end of video stream
        cv::imshow("image", frame);
        cv::namedWindow("image",WINDOW_NORMAL);
        cv::resizeWindow("image", 600,600);
        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}



