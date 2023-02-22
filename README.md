# YOLOtalk-GUI install tutorial


## Step 0. 建立虛擬環境(Optional) 
```code=
python3 -m venv ~/my-example-virtual-environment-please-change-this-name
source ~/my-example-virtual-environment-please-change-this-name/bin/activate
pip install -U pip wheel
```

## Step 1. 下載 YOLOtalk-GUI 

```bash=
git clone https://github.com/IoTtalk/YOLOtalk-GUI.git
```
**專案架構為**
```
----- YOLOTalk-GUI
  |
  |-- libs 
  |    |
  |    |--YOLO_SSIM.py
  |    |--DAN.py
  |    |--utils.py
  |    |--SSIM_utils.py
  |
  |--darknet  
  |--multi-object-tracker
  |--weights
  |--cfg_person
  |--YOLOTalk-GUI ★★★
       |
       |--YOLOTalk.py  (Flask)
       |--config.py
       |--templates
       |--static
            |--Json_Info
            |--alias_pict
            |--record
```                        
## Step 2. 編譯 darknet
進入==darknet==資料夾，並根據本身電腦環境，編輯==Makefile==檔案
參考: https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make
```code=
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
接著進行編譯
```code=
make
```

:::danger
若出現
```code= 
/bin/sh: 1: nvcc: not found
Makefile:185: recipe for target 'obj/convolutional_kernels.o' failed
make: *** [obj/convolutional_kernels.o] Error 127
make: *** Waiting for unfinished jobs....
```
請使用以下指令指定 nvcc 路徑
它被裝在 CUDA 資料夾 /usr/local/cuda 下。請修改 PATH。
```code=
export PATH=$PATH:/usr/local/cuda/bin
```
並再重新編譯。

```code=
make clean
make -j4
:::
 
## Step 3. 安裝套件 
```code=
pip install -r requirements.txt
```

## Step 4. 設置 config.py flask物件參數
1.host、port 參數若有需要更改請至config.py。
```code=
Config ={
    "DEBUG" : False,
    "use_reloader" : False,
    "host" : "127.0.0.1",
    "port" : "5000",
}
```
## Step 5. Quick start
```=
cd ../YOLOtalk_GUI
python YOLOtalk.py
```

## 資料夾說明

==cfg_person==

1.coco.data : 放置指定的資料(classes、trainset、validset、names、backup)
2.放置 v4、v7 各個模型 cfg 檔案

==weights==

1.放置 v4-tiny、v7-tiny 權重檔案、訓練檔案
2.若想使用 yolov4、v7 完整版可使用下面指令獲取檔案
```code=
cd ../weights
# v4
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.conv.137
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights
# v7
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.conv.132
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.weights
```



==YOLOTalk-GUI==
1. YOLOTalk-GUI/static/Json_Info  : 放置各個圍籬紀錄參數檔案
2. YOLOTalk-GUI/static/alias_pict : 放置各個圍籬所擷取圖片，用於 plotarea 功能繪製圍籬的底圖
3. YOLOTalk-GUI/static/record     : 放置 YOLO_SSIM.py 所記錄之照片、圖片

###### tags: `YOLOTalk` 