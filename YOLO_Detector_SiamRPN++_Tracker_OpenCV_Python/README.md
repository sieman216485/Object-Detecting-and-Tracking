# YOLO Detector and SiamRPN++ tracker

## Detector
 - `YOLOv5s`
 - `YOLOv8s`

## Tracker
 - `SiamRPN++` tracker

## Testing

To test `SiamRPN++`, it needs to unzip `search_net.onnx.zip` and `target_net.onnx.zip`.

`ROI` tracking

```bash
python main_YOLO.py
```

SOT by clicking an object

```bash
python main.py
```