/**
 * \file VideoStabilizer.hpp
 * \brief Module for kallman based video Stabilization
 * \author Manuel Deneu Claude Perdigou
 * \version 0.1
 * \date 21/05/2016
 *
 *  Public HEADER
 *
 */

/*
 Orignal code:
 Thanks Nghia Ho for his excellent code.
 And,I modified the smooth step using a simple kalman filter .
 So, it can processes live video streaming.
 modified by chen jia.
 email:chenjia2013@foxmail.com
 */

#ifndef VideoStabilizer_hpp
#define VideoStabilizer_hpp

#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>



// This video stablisation smooths the global trajectory using a sliding average window

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

/*
 Basic Use:
 
 VideoStabilizer stabilizer;
 
 cv::Mat matToTrack; // maybe from VideoCapture
 
 if(stabilizer.iteration(matToTrack  )
 {
 cv::imshow("My Win", stabilizer.getStabilizedOutput()  );
 
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
 
 */

namespace VK
{
    class VideoStabilizer
    {
        /* **** **** **** **** **** **** **** **** **** **** **** **** **** */
        struct TransformParam
        {
            TransformParam() {}
            TransformParam(double _dx, double _dy, double _da) {
                dx = _dx;
                dy = _dy;
                da = _da;
            }
            
            double dx;
            double dy;
            double da; // angle
        };
        
        struct Trajectory
        {
            Trajectory() {}
            Trajectory(double _x, double _y, double _a) {
                x = _x;
                y = _y;
                a = _a;
            }
            // "+"
            friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
                return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
            }
            //"-"
            friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
                return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
            }
            //"*"
            friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
                return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
            }
            //"/"
            friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
                return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
            }
            //"="
            Trajectory operator =(const Trajectory &rx){
                x = rx.x;
                y = rx.y;
                a = rx.a;
                return Trajectory(x,y,a);
            }
            
            double x;
            double y;
            double a; // angle
        };
        
        /* **** **** **** **** **** **** **** **** **** **** **** **** **** */
    public:
        enum  { HORIZONTAL_BORDER_CROP = 20 };
        
        
        typedef struct
        {
            double dx;
            double dy;
            double da;
        } TransformState;
        
        typedef struct
        {
            double x;
            double y;
            double a;
        }  TransformTrajectory;
        
        VideoStabilizer();
        
        /*
         Warning : Currently only RGV images are working as input.
         This needs serious optimizations since all frames are converted to grayscale buffers inside.
         For now you can use `cvtColor( mat, mat2, CV_GRAY2RGB)` or something to convert input frame as RGB.
         */
        int iteration( const cv::Mat &frame );
        
        
        void resetTrajectoryTo( float x , float y , float a)
        {
            _tTrajectory.x = (double) x;
            _tTrajectory.y = (double) y;
            _tTrajectory.a = (double) a;
            
        }
        
        const cv::Mat &getStabilizedOutput() const /*noexcept*/
        {
            return _stabilizedOutput;
        }
        const TransformState &getCurrentTransform() const /*noexcept*/
        {
            return _tState;
        }
        const TransformTrajectory &getTrajectory() const /*noexcept*/
        {
            return _tTrajectory;
        }
        const TransformTrajectory &getSmoothedTrajectory() const /*noexcept*/
        {
            return _tSmoothedTrajectory;
        }
        
        int getFrameNumber() const /*noexcept*/
        {
            return _frameNum;
        }
        
    private:
        void init( const cv::Mat &firstFrame);
        
        
        cv::Mat cur_grey;
        
        cv::Mat _stabilizedOutput;
        
        cv::Mat prev;
        cv::Mat prev_grey;
        
        cv::Mat last_T;
        cv::Mat prev_grey_;
        cv::Mat cur_grey_;
        
        std::vector <TransformParam> prev_to_cur_transform; // previous to current
        
        TransformState _tState;
        TransformTrajectory _tTrajectory;
        TransformTrajectory _tSmoothedTrajectory;
        
        std::vector <Trajectory> trajectory; // trajectory at all frames
        
        std::vector <Trajectory> smoothed_trajectory; // trajectory at all frames
        Trajectory X;//posteriori state estimate
        Trajectory	X_;//priori estimate
        Trajectory P;// posteriori estimate error covariance
        Trajectory P_;// priori estimate error covariance
        Trajectory K;//gain
        Trajectory	z;//actual measurement
        double pstd;
        double cstd;
        Trajectory Q;
        Trajectory R;
        
        std::vector <TransformParam> new_prev_to_cur_transform;
        
        cv::Mat T;
        
        int vert_border;
        
        int _frameNum;
        
    };
};
#endif /* VideoStabilizer_hpp */
