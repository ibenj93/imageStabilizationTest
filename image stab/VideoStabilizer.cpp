/**
 * \file VideoStabilizer.cpp
 * \brief Module for kallman based video Stabilization
 * \author Manuel Deneu Claude Perdigou
 * \version 0.1
 * \date 21/05/2016
 *
 *
 */

#include <cmath>
#include <stdio.h>
#include "VideoStabilizer.hpp"


VK::VideoStabilizer::VideoStabilizer():
_tTrajectory ( { 0.,0.,0.} ),
_tSmoothedTrajectory ( { 0.,0.,0.}) ,

_tState ( { 0.,0.,0.} ),


pstd ( 4e-3 ),
cstd ( 0.25 ),

Q ( Trajectory(pstd,pstd,pstd) ),
R ( Trajectory(cstd,cstd,cstd) ),
T ( cv::Mat(2,3,CV_64F) ),

vert_border ( -1 ), // invalid state
_frameNum ( 0 )
{
    
    
}

void VK::VideoStabilizer::init( const cv::Mat &firstFrame)
{
    vert_border = HORIZONTAL_BORDER_CROP * firstFrame.rows / firstFrame.cols; // get the aspect ratio correct
    
    cvtColor(firstFrame, prev_grey, cv::COLOR_BGR2GRAY);
    
    firstFrame.copyTo(prev);
}


int VK::VideoStabilizer::iteration( const cv::Mat &frame )
{
    if( _frameNum == 0) // fist time
    {
        init( frame);
    }
    
    if(frame.data == NULL)
    {
        assert( false);
        return 0;
    }
    
    cvtColor(frame, cur_grey, cv::COLOR_BGR2GRAY);
    
    
    // vector from prev to cur
    std::vector <cv::Point2f> prev_corner, cur_corner;
    std::vector <cv::Point2f> prev_corner2, cur_corner2;
    std::vector <uchar> status;
    std::vector <float> err;
    
    goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
    calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);
    
    // weed out bad matches
    for(size_t i=0; i < status.size(); i++)
    {
        if(status[i])
        {
            prev_corner2.push_back(prev_corner[i]);
            cur_corner2.push_back(cur_corner[i]);
        }
    }
    
    // translation + rotation only
    try
    {
        T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing
    }
    catch (const cv::Exception& e)
    {
        printf("Caught exception cv::Exception %s \n" , e.what() );
        return 0;
    }
    
    
    // in rare cases no transform is found. We'll just use the last known good transform.
    if(T.data == NULL)
    {
        last_T.copyTo(T);
    }
    
    T.copyTo(last_T);
    
    // decompose T
    _tState.dx = T.at<double>(0,2);
    _tState.dy = T.at<double>(1,2);
    _tState.da = atan2(T.at<double>(1,0), T.at<double>(0,0));
    //
    //prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
    
    //out_transform << k << " " << dx << " " << dy << " " << da << std::endl;
    //
    // Accumulated frame to frame transform
    _tTrajectory.x += _tState.dx;
    _tTrajectory.y += _tState.dy;
    _tTrajectory.a += _tState.da;
    
    //out_trajectory << k << " " << x << " " << y << " " << a << std::endl;
    
    z = Trajectory(_tTrajectory.x,
                   _tTrajectory.y,
                   _tTrajectory.a
                   );
    
    if( _frameNum == 0 )
    {
        // intial guesses
        X = Trajectory(0,0,0); //Initial estimate,  set 0
        P =Trajectory(1,1,1); //set error variance,set 1
    }
    else
    {
        
        X_ = X; //X_(k) = X(k-1);
        P_ = P+Q; //P_(k) = P(k-1)+Q;
        
        K = P_/( P_+R ); //gain;K(k) = P_(k)/( P_(k)+R );
        X = X_+K*(z-X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k));
        P = (Trajectory(1,1,1)-K)*P_; //P(k) = (1-K(k))*P_(k);
    }
    
    _tSmoothedTrajectory.x = X.x;
    _tSmoothedTrajectory.y = X.y;
    _tSmoothedTrajectory.a = X.a;
    
    double diff_x = X.x - _tTrajectory.x;
    double diff_y = X.y - _tTrajectory.y;
    double diff_a = X.a - _tTrajectory.a;
    
    _tState.dx += diff_x;
    _tState.dy += diff_y;
    _tState.da += diff_a;
    
    //new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
    //
    //out_new_transform << k << " " << dx << " " << dy << " " << da << std::endl;
    //
    
    T.at<double>(0,0) = cos( _tState.da );
    T.at<double>(0,1) = -sin( _tState.da );
    T.at<double>(1,0) = sin( _tState.da );
    T.at<double>(1,1) = cos( _tState.da );
    
    T.at<double>(0,2) = _tState.dx;
    T.at<double>(1,2) = _tState.dy;
    
    
    
    warpAffine(prev, _stabilizedOutput, T, frame.size());
    
    _stabilizedOutput = _stabilizedOutput(cv::Range(vert_border, _stabilizedOutput.rows-vert_border),
                                          cv::Range(HORIZONTAL_BORDER_CROP, _stabilizedOutput.cols-HORIZONTAL_BORDER_CROP));
    
    // Resize cur2 back to cur size, for better side by side comparison
    resize(_stabilizedOutput, _stabilizedOutput, frame.size());
    
    
    
    
    prev = frame.clone();
    cur_grey.copyTo(prev_grey);
    
    _frameNum++;
    
    
    
    return 1;
}
