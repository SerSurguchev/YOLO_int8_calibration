# Post-Training Quantization Using Calibration.

1) This project is based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5). Place install it first.

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
2) Place tensorrt_quat.py, calibrator.py, util_trt.py in the home directory ( /yolov5/ ).
3) Select 1000 - 2000 random images from dataset to run calibration

4) Export YOLO trained .pt weighs file from Pytorch to ONNX and Engine file
```bash
python export.py --weights yolov5n.pt --include engine --half     --imgsz 640
                                                       --simplify #onnx-simplifier (https://github.com/daquexian/onnx-simplifier)
```
5) Run calibration
```bash
python3 tensorrt_quat.py --use_int8 --h 640 --w 640

```
