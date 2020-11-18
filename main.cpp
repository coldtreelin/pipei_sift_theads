#include "opencv2/opencv_modules.hpp"
# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "xfeatures2d\nonfree.hpp"
 
#include <stdio.h>
#include <vector>
#include <iostream>
using namespace cv;
using namespace xfeatures2d;
using namespace std;
//void readme();
 
/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{ 
  Mat img_1 = imread( "baby.jpg", CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( "tmp.jpg", CV_LOAD_IMAGE_GRAYSCALE );
  //if( !img_1.data || !img_2.data )
  //{ printf(" --(!) Error reading images \n"); return -1; }
 
  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;
  typedef SURF Feature;
  auto detector = Feature::create(minHessian);
 
  std::vector<KeyPoint> keypoints_1, keypoints_2;
 
  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );
 
  //-- Step 2: Calculate descriptors (feature vectors)
  auto extractor =  Feature::create();
 
  Mat descriptors_1, descriptors_2;

  extractor->compute( img_1, keypoints_1, descriptors_1 );
  extractor->compute( img_2, keypoints_2, descriptors_2 );
  // cout<<descriptors_1;
   auto t = keypoints_1[0];
   for(int i=0;i<keypoints_1.size();i++)
   {
      t = keypoints_1[i];
      cout<<i+1<<":"<<"angle:"<<t.angle<<" "<<"id:"<<t.class_id<<" "<<"octave:"<<t.octave<<" "<<"reps:"<<t.response<<" "<<"size:"<<t.size<<endl;
   }

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );
 
  double max_dist = 0; double min_dist = 100;
 
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  //for( int i = 0; i <matches.size(); i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //cout<<matches.size()<<" "<<descriptors_1.rows<<" "<<descriptors_1.cols<<endl;
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
 
  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;
 
  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }
 
  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
 
  //-- Show detected matches
  imshow( "Good Matches", img_matches );
 
  for( int i = 0; i < (int)good_matches.size(); i++ )
  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
 
  waitKey(0);
  system("pause");
  return 0;
}
 
/**
 * @function readme
 */
void readme()
{ 
  printf(" Usage: ./SURF_FlannMatcher <img1> <img2>\n"); 
}
 
