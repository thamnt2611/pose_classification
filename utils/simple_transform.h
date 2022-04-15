#include <vector>
#include <yolo_v2_class.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/hal/interface.h>
#include <torch/script.h>
#include <pose_type.h>

// typedef struct{
//     int x;
//     int y;
// }point_t;

void test_transformation(const cv::Mat& img, 
                        const bbox_t& bbox, 
                        at::Tensor& inp, 
                        bbox_t& n_bbox,
                        const scale_t& target_scale,
                        float scale_mult = 1.25);

void get_simple_affine_transformation(cv::Mat& transform_mat, 
                        const cv::Point2f& center,
                        const scale_t& orig_scale,
                        const scale_t& target_scale, 
                        bool inv);

void heatmap_to_keypoints(const at::Tensor& heatmap,
                    const bbox_t& box,
                    at::Tensor& kp_coords, // shape : numjoints * 2
                    at::Tensor& kp_scores);

void mat_to_tensor(const cv::Mat& mat, at::Tensor& tensor);
void tensor_to_mat(const at::Tensor& tensor, cv::Mat& mat);