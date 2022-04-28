import numpy as np
import cv2
import torch
from utils.transform import get_affine_transform


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


class SimpleTransform(object):
    def __init__(self, input_size):
        self._input_size = input_size
        self._aspect_ratio = float(input_size[1]) / input_size[0]

    def test_transformation(self, src, bbox):
        center_x, center_y, w, h = bbox
        center, scale = self._box_to_center_scale(
            center_x, center_y, w, h, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = self._center_scale_to_box(center, scale)
        img = self._im_to_torch(img)

        # Thực hiện normalization (trừ đi mean)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        return img, bbox

    def _im_to_torch(self, img):
        """Transform ndarray image to torch tensor.
        Parameters
        ----------
        img: numpy.ndarray
            An ndarray with shape: `(H, W, 3)`
        Returns
        -------
        torch.Tensor
            A tensor with shape: `(3, H, W)`.
        """
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = to_torch(img).float()
        if img.max() > 1:
            img /= 255
        return img

    def _box_to_center_scale(self, center_x, center_y, w, h, aspect_ratio=1.0, scale_mult=1.25):
        """
            box -> dieu chinh aspect_ratio(theo chieu lon hon) -> phong to, dua vao scale_mult
            scale = (W, H)
        """
        pixel_std = 1 # TODO: khong hieu
        center = np.zeros((2), dtype = np.float32)
        center[0] = center_x
        center[1] = center_y

        # import pdb; pdb.set_trace()
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = aspect_ratio * h
        scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
        if center[0] != -1: # TODO: khong hieu
            scale = scale * scale_mult
        return center, scale

    def _center_scale_to_box(self, center, scale):
        pixel_std = 1.0
        w = scale[0] * pixel_std
        h = scale[1] * pixel_std
        xmin = center[0] - w * 0.5
        ymin = center[1] - h * 0.5
        xmax = xmin + w
        ymax = ymin + h
        bbox = [xmin, ymin, xmax, ymax]
        # print("BBOX: ", center[0], center[1], w, h)
        return bbox
        