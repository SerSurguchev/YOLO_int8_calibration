# Post-Training Quantization Using Calibration.

This project is based on [ultralytics/yolov5](https://github.com/ultralytics/yolov5). Place install it first.

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

Place tensorrt_quat.py, calibrator.py, util_trt.py in the home directory ( /yolov5/ ).

Export YOLO trained .pt weighs file from Pytorch to ONNX and Engine file

```bash
python export.py --weights yolov5n.pt --include engine --half --imgsz 640
                                                       --simplify
```

```bash
python3 tensorrt_quat.py --use_int8

```
