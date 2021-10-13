# Tracking Pose from multiple Videos 

In this project I used mmpose library's top_down_pose_tracking_demo_with_mmdet.py to track pose from multiple videos after merging them

* Watch the output video here
* See the full code [here](https://github.com/shoaib6174/Tracking-Pose-from-Videos/blob/main/top_down_pose_tracking_demo_with_mmdet.py)
* See the code for extracting the frames parallaly and merging them to make a video [here](https://github.com/shoaib6174/Tracking-Pose-from-Videos/blob/main/merge_videos.py)


I have tested this code on Google Colab. You can use this notebook to run the code. 


To run this code locally clone this repo-
```
git clone https://github.com/shoaib6174/Tracking-Pose-from-Videos.git

cd Tracking-Pose-from-Videos
```
Run the following code to install all the dependencies
```
pip install torch
pip install wheel
pip install mmcv-full
pip install mmdet

# clone mmpose repo
rm -rf mmpose
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose

# install mmpose dependencies
pip install -r requirements.txt

# install mmpose in develop mode
pip install -e .

#return to Tracking-Pose-from-Videos directory
cd ..

pip install opencv-python
pip install mmtrack
```

Then to run the top_down_pose_tracking_demo_with_mmdet.py run the following command- 
```
python top_down_pose_tracking_demo_with_mmdet.py \
    mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --input_videos_folder_path input_videos \
    --out-video-root vis_results
```

If you get any error of missing dependencies please install them. 

I have tested this code on Google Colab. You can use this notebook to run the code. 
