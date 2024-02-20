YOLOv8 implementation without [DFL](https://ieeexplore.ieee.org/document/9792391) using PyTorch

### Installation

```
conda create -n YOLO python=3.8
conda activate YOLO
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```

### Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

| Version | Epochs | Box mAP |                   Download |
|:-------:|:------:|--------:|---------------------------:|
|  v8_n   |  500   |    37.0 | [model](./weights/best.pt) |
|  v8_n*  |  500   |    37.3 |                          - |
|  v8_s   |  500   |       - |                          - |
|  v8_s*  |  500   |    44.9 |                          - |
|  v8_m   |  500   |       - |                          - |
|  v8_m*  |  500   |    50.2 |                          - |
|  v8_l   |  500   |       - |                          - |
|  v8_l*  |  500   |    52.9 |                          - |
|  v8_x   |  500   |       - |                          - |
|  v8_x*  |  500   |    53.9 |                          - |

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.529
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.401
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.408
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.585
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.764
```

* `*` means that it is from original repository, see reference
* In the official YOLOv8 code, mask annotation information is used, which leads to higher performance

### Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

#### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/ultralytics/ultralytics
