#ifndef POSE_TYPES
#define POSE_TYPES
#include<darknet.h>
#include<vector>
typedef struct{
    int w;
    int h;
}scale_t;

typedef struct{
    float scaleX;
    float scaleY;
}scale_factor_t;

typedef struct kp_t{
    int x {};
    int y {};
    float prob {};
}kp_t;

typedef struct point_t{
    float x {};
    float y {};
}point_t;

// THAY DOI SO NUM_JOINT O DAY
typedef struct pose_t{
    std::vector<kp_t> keypoints{26}; 
    float score {};
    bbox_t bounding_box;
}pose_t;

#endif