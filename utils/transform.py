import numpy as np
import cv2

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def perspective_transform(point, trans_matrix):
    new_point = np.ones((2))
    denominator = trans_matrix[2][0] * point[0] + trans_matrix[2][1] * point[1] + trans_matrix[2][2] 
    print(trans_matrix.shape)
    new_point[0] = (trans_matrix[0][0] * point[0] + trans_matrix[0][1] * point[1] + trans_matrix[0][2]) / denominator
    new_point[1] = (trans_matrix[1][0] * point[0] + trans_matrix[1][1] * point[1] + trans_matrix[1][2]) / denominator
    return new_point

def get_dir(src_point, rot_rad):
    # OK
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    # OK (điểm trả về, a và b sẽ tạo thành tam giác vuông)
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                        scale,
                        rot,
                        output_size,
                        shift=np.array([0, 0], dtype=np.float32),
                        inv=0):
    """
        Phép biến đổi ở đây tương đương với resize (?)
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def transform_hm_to_bbox(pred_point, center, scale, hm_size):
    target_coords = np.zeros(pred_point.shape)
    # chu y : inv = 1
    trans = get_affine_transform(center, scale, 0, hm_size, inv=1) 
    target_coords[0:2] = affine_transform(pred_point[0:2], trans)
    return target_coords

def heatmap_to_coord_simple(hm, bbox):
    # phai chuyen truoc khi sd ham cua np tren tensor
    if not isinstance(hm,np.ndarray):
        hm = hm.cpu().data.numpy()

    # lấy vị trí, giá trị pixel có heat map cao nhất
    kp_coords, kp_probs = get_max_pred(hm)

    # nhich kp them theo huong tang cua heatmap
    for i in range(kp_coords.shape[0]):
        hm_i = hm[i]
        hm_h, hm_w = hm_i.shape[0], hm_i.shape[1]
        px = int(kp_coords[i][0])
        py = int(kp_coords[i][1])
        if(1 < px < hm_w-1) and (1 < py < hm_h-1):
            diff = np.array((hm_i[py][px + 1] - hm_i[py][px-1], 
                            hm_i[py+1][px] - hm_i[py-1][px]))
            kp_coords += np.sign(diff) * 0.25

    # transform coordinate from heatmap space to bbox space
    xmin, ymin, xmax, ymax = bbox
    w = xmax -xmin
    h = ymax - ymin
    scale = np.array([w, h])
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    preds = np.zeros_like(kp_coords)
    # print("KP COORDS - HM")
    # print(kp_coords)
    for i in range(len(kp_coords)):
        preds[i] = transform_hm_to_bbox(kp_coords[i], center, scale,
                                   [hm_w, hm_h])
    # print("PREDS")
    # print(preds)
    return preds, kp_probs

def get_max_pred(hms):
    num_joints = hms.shape[0]
    width = hms.shape[2]
    # NOTE
    reshaped_hms = hms.reshape(num_joints, -1)
    preds = np.argmax(reshaped_hms, 1)
    pred_vals = np.max(reshaped_hms, 1)

    preds = preds.reshape((num_joints, 1))
    pred_vals = pred_vals.reshape((num_joints, 1))

    preds = np.tile(preds, (1, 2)).astype(np.float32)
    preds[:, 0] = preds[:, 0] % width # x
    preds[:, 1] = np.floor(preds[:, 1] / width) # y

    pred_mask = np.tile(np.greater(pred_vals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask
    # print("PREDS")
    # print(preds)
    return preds, pred_vals

def get_custome_transform_matrix():
        """
          Tính toán ma trận biến đổi để chuyển video thành góc quay thẳng đứng từ trên xuống
        """
        # im = cv2.imread("inputs/many_people_shot.jpg")
        src = np.zeros((4, 2), dtype=np.float32)
        dst = np.zeros((4, 2), dtype=np.float32)
        src[0, :] = np.array([1191.0, 120.0])
        src[1, :] = np.array([1580.0, 833.0])
        src[2, :] = np.array([386.0, 132.0])
        src[3, :] = np.array([17.0, 866.0])

        dst[0, :] = np.array([1600.0, 0])
        dst[1, :] = np.array([1600.0, 900.0])
        dst[2:, :] = np.array([0.0, 0.0])
        dst[3:, :] = np.array([0.0, 900.0])

        trans = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        # w = frame.shape[1]
        # h = frame.shape[0]
        # print(w, h)
        # n_im = cv2.warpPerspective(frame, trans, (w, h), flags=cv2.INTER_LINEAR)
        # cv2.imshow("Image", n_im)
        # cv2.waitKey(0)
        return trans

if __name__=="__main__":
    im = cv2.imread("inputs/many_people_shot.jpg")
    src = np.zeros((4, 2), dtype=np.float32)
    dst = np.zeros((4, 2), dtype=np.float32)
    src[0, :] = np.array([1191.0, 120.0])
    src[1, :] = np.array([1580.0, 833.0])
    src[2, :] = np.array([386.0, 132.0])
    src[3, :] = np.array([17.0, 866.0])

    dst[0, :] = np.array([1600.0, 0])
    dst[1, :] = np.array([1600.0, 900.0])
    dst[2:, :] = np.array([0.0, 0.0])
    dst[3:, :] = np.array([0.0, 900.0])

    trans = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    w = im.shape[1]
    h = im.shape[0]
    # print(w, h)
    n_im = cv2.warpPerspective(im, trans, (w, h), flags=cv2.INTER_LINEAR)
    cv2.imshow("Image", n_im)
    cv2.waitKey(0)