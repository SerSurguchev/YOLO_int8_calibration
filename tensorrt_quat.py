import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob
import os
import cv2
import argparse


def preprocess_v1(image_raw, height, width):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    return np.transpose(image, [2, 0, 1])


class DataLoader:
    def __init__(self, batch, batch_size, calib_img_dir, height, width):
        self.index = 0
        self.batch_size = batch_size
        self.length = batch
        self.height = height
        self.width = width

        self.img_list = glob.glob(os.path.join(calib_img_dir, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(
            CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'

        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess_v1(img, self.height, self.width)
                self.calibration_data[i] = img

            self.index += 1

            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=100, help='Model quantization times')
    parser.add_argument('--batch_size', type=int, default=1, help='How many pictures are input')
    parser.add_argument('--h', type=int, default=640, help='Image height')
    parser.add_argument('--w', type=int, default=640, help='Image width')
    parser.add_argument('--calib_img_dir', type=str, default='samples', help='Calibration data path')
    parser.add_argument('--onnx_model_path', type=str, default='best.onnx',
                        help='Onnx weights file path')
    parser.add_argument('--use_fp16', action='store_true', help='Use fp16 weights')
    parser.add_argument('--use_int8', action='store_true', help='Use int8 weights calibration')
    parser.add_argument('--engine_model_path', type=str, default='best_int8.engine',
                        help='Trt_engine file')
    parser.add_argument('--calibration_table', type=str, default='yolov5n_calibration.cache',
                        help='Calibration data')
    return parser.parse_args()


def main():
    opt = get_args()
    fp16_mode, int8_mode = opt.use_fp16, opt.use_int8
#    fp16_mode, int8_mode = False, True
    print('*** Onnx to tensorrt begin ***')
    # Calibration
    calibration_stream = DataLoader(
        opt.batch,
        opt.batch_size,
        opt.calib_img_dir,
        opt.h,
        opt.w
    )

    engine_fixed = util_trt.get_engine(
        opt.batch_size,
        opt.onnx_model_path,
        opt.engine_model_path,
        fp16_mode=fp16_mode,
        int8_mode=int8_mode,
        calibration_stream=calibration_stream,
        calibration_table_path=opt.calibration_table,
        save_engine=True
    )

    assert engine_fixed, 'Broken engine_fixed'
    print('*** Onnx to tensorrt completed ***\n')


if __name__ == '__main__':
    main()
