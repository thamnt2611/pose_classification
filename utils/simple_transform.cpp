#include "simple_transform.h"
#include "pose_type.h"
#include<torch/script.h>
#include <stdio.h> 

cv::Point2f get_3rd_point(cv::Point2f p1, cv::Point2f p2){
    cv::Point2f line_dir = p2 - p1;
    return p1 + cv::Point2f{-line_dir.y, line_dir.x};
}

void get_simple_affine_transformation(cv::Mat& transform_mat, 
                        const cv::Point2f& center,
                        const scale_t& orig_scale,
                        const scale_t& target_scale, 
                        bool inv){
    cv::Point2f src_tri[3];
    cv::Point2f dst_tri[3];

    // center cua src box
    src_tri[0] = cv::Point2f(center.x, center.y);
    // center cua target box
    dst_tri[0] = cv::Point2f(target_scale.w * 0.5, target_scale.h * 0.5);

    cv::Point2f src_dir{orig_scale.w / 2.0f, 0.0f};
    src_tri[1] = src_tri[0] + src_dir;
    cv::Point2f dst_dir{target_scale.w / 2.0f, 0.0f};
    dst_tri[1] = dst_tri[0] + dst_dir;

    src_tri[2] = get_3rd_point(src_tri[0], src_tri[1]);
    dst_tri[2] = get_3rd_point(dst_tri[0], dst_tri[1]);

    if (!inv){
        transform_mat = cv::getAffineTransform(src_tri, dst_tri);
    }
    else{
        transform_mat = cv::getAffineTransform(dst_tri, src_tri);
    }
}

void test_transformation(const cv::Mat& img, 
                        const bbox_t& bbox, 
                        at::Tensor& inp, 
                        bbox_t& n_bbox,
                        const scale_t& target_scale,
                        float scale_mult){
    
        // Gồm các bước:
        // - cho ảnh gốc cho về cùng tỉ lệ với ảnh đích (phong to)
        // - resize về kích thước đích (affine transformation)
        // - trả về: 
        //     + ảnh đã đc resize
        //     + bbox - tương ứng với ảnh đã resize (center ko đổi)
        //     +      - hay là thể hiện scale ban đầu

    cv::Point2f center {bbox.x, bbox.y};
    float w = bbox.w;
    float h = bbox.h;
    float aspect_ratio = target_scale.w * 1.0 / target_scale.h;
    if (w > h * aspect_ratio){
        h = w / aspect_ratio;
    }
    else if (h > w / aspect_ratio){
        w = h * aspect_ratio;
    }
    scale_t orig_scale{w, h};
    cv::Mat trans_mat(2, 3, CV_32FC1);
    get_simple_affine_transformation(trans_mat, center, orig_scale, target_scale, false);
    cv::Mat transformed_mat;
    cv::warpAffine(img, transformed_mat, trans_mat, cv::Size {target_scale.w, target_scale.h}, cv::INTER_LINEAR);
    
    cv::Mat normalized_mat;
    transformed_mat.convertTo(normalized_mat, CV_32F, 1.0 / 255);
    
    cv::Mat channels[3];
    cv::split(normalized_mat, channels);
    channels[0] = (channels[0] - 0.406);
    channels[1] = (channels[1] - 0.457);
    channels[2] = (channels[2] - 0.480);

    cv::merge(channels, 3, normalized_mat);
    inp = torch::from_blob(normalized_mat.data, 
                        {1, normalized_mat.rows, normalized_mat.cols, normalized_mat.channels()},
                        torch::TensorOptions(torch::kFloat))
                        .permute({0, 3, 1, 2})
                        .cuda();
    // tại sao center của bbox ko đổi ?
    n_bbox = bbox;
    n_bbox.w = w * scale_mult;
    n_bbox.h = h * scale_mult;
}

void get_max_pred(const at::Tensor& heatmap,
                at::Tensor& kp_coords, // shape : numjoints * 2
                at::Tensor& kp_scores){   // shape : numjoints 
    int num_joints = heatmap.sizes()[0];
    int width = heatmap.sizes()[2];
    at::Tensor reshaped_hm = heatmap.view({num_joints, -1});
    at::Tensor preds = reshaped_hm.argmax(1);
    auto accessor_a = kp_scores.accessor<float, 1>();
    for(int i = 0; i < num_joints; i++){
        int max_idx = preds[i].item<int>();
        accessor_a[i] = reshaped_hm[i][max_idx].item().to<float>();
    }

    auto accessor_b = kp_coords.accessor<float, 2>();
    for(int i = 0; i < num_joints; i++){
        float score = kp_scores[i].item().to<float>();
        if (score > 0){
            accessor_b[i][0] = preds[i].item().to<int>() % width;
            accessor_b[i][1] = std::floor(preds[i].item().to<int>() / width);
        }
        else{
            accessor_b[i][0] = 0;
            accessor_b[i][1] = 0;
        }
    }
}

void transform_keypoint(const point_t& src_point,
                    point_t& dst_point,
                    const scale_t& src_scale,
                    const point_t& dst_center,
                    const scale_t& dst_scale){
    cv::Mat trans_matrix(2, 3, CV_32FC1);
    cv::Point2f n_dst_center{dst_center.x, dst_center.y};
    get_simple_affine_transformation(trans_matrix, n_dst_center, dst_scale, src_scale, true);

    float data[3];
    data[0] = src_point.x; data[1] = src_point.y; data[2] = 1.0;
    cv::Mat src_mat {1, 3, CV_32FC1, data};

    dst_point.x = trans_matrix.at<double>(0, 0) * src_point.x + trans_matrix.at<double>(0, 1) * src_point.y + trans_matrix.at<double>(0, 2);
    dst_point.y = trans_matrix.at<double>(1, 0) * src_point.x + trans_matrix.at<double>(1, 1) * src_point.y + trans_matrix.at<double>(1, 2);
}

void fine_tune_kp(const at::Tensor& heatmap, 
                at::Tensor& kp_coords){
    int hm_height = heatmap.sizes()[1];
    int hm_width = heatmap.sizes()[2];
    auto accessor = kp_coords.accessor<float, 2>();
    for (int i = 0; i < kp_coords.sizes()[0]; i++){
        int kp_x = kp_coords[i][0].item().to<int>(); int kp_y = kp_coords[i][1].item().to<int>();
        if ((kp_x > 1) && (kp_x < hm_width - 1) && (kp_y > 1) && (kp_y < hm_height - 1)){
            float x_right = heatmap[i][kp_y][kp_x + 1].item<float>();
            float x_left = heatmap[i][kp_y][kp_x - 1].item<float>();
            if (x_right > x_left){
                accessor[i][0] += 0.25;
            }
            else{
                accessor[i][0] -= 0.25;
            }

            if (heatmap[i][kp_y + 1][kp_x].item().to<float>() > heatmap[i][kp_y - 1][kp_x].item().to<float>()){
                accessor[i][1] += 0.25;
            }
            else{
                accessor[i][1] -= 0.25;
            }
        }
    }
}

void heatmap_to_keypoints(const at::Tensor& heatmap,
                    const bbox_t& box,
                    at::Tensor& kp_coords, // shape : numjoints * 2
                    at::Tensor& kp_scores){
    int num_joints = heatmap.sizes()[0];

    get_max_pred(heatmap, kp_coords, kp_scores);

    fine_tune_kp(heatmap, kp_coords);

    scale_t scale {box.w, box.h};
    point_t center {box.x, box.y};
    auto accessor = kp_coords.accessor<float, 2>();
    scale_t heatmap_scale {heatmap.sizes()[2], heatmap.sizes()[1]};
    for (int i = 0; i < num_joints; i++){
        point_t src_point {kp_coords[i][0].item().to<float>(), kp_coords[i][1].item().to<float>()};
        point_t dst_point;
        transform_keypoint(src_point, dst_point, heatmap_scale, center, scale);
        accessor[i][0] = dst_point.x;
        accessor[i][1] = dst_point.y;
    }

}

void tensor_to_mat(const at::Tensor& tensor, cv::Mat& mat){
    std::memcpy((void*) mat.data, tensor.data_ptr(), sizeof(torch::kU8)*tensor.numel()); // sua lai kieu du lieu
}
