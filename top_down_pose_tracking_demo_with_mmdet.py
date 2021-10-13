# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import re
import numpy as np
from os.path import isfile, join
import time
import concurrent.futures
import cv2

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# My code starts   

# Function for extracting the frames of 4 videos 

def frame_capture(path,video_no):
    print("Extracting frames of video no-%d" % video_no)
    vidcap = cv2.VideoCapture(path)
    os.makedirs("frames/"+str(video_no) , exist_ok=True)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite("frames/%d/frame%d.jpg" % (video_no, count), image)   # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 0.0336
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    return  "Finished Frame Extraction of video no-" +str(video_no)


# function for mergeing frames of 4 videos in 2x2 grid

def make_frame_grid(image_paths,grid_no):
    
    im1 = cv2.imread(image_paths[0])
    im2 = cv2.imread(image_paths[1])
    im3 = cv2.imread(image_paths[2])
    im4 = cv2.imread(image_paths[3])
    
    im_v1 = cv2.vconcat([im1, im2])
    im_v2 = cv2.vconcat([im3, im4])
    im_h = cv2.hconcat([im_v1, im_v2])
    name = "grid/%d.jpg" % grid_no
    img = cv2.resize(im_h, (1920, 1080))
    cv2.imwrite(name, img)
 

# function for making a video from grid images

def make_video_from_grid_images(pathIn, pathOut, fps):
    #get file location
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    #reading each files
    for i in range(len(files)):
        filename=pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)

        #inserting the frames into an image array
        frame_array.append(img)
    
    #creating the video
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

# function for combining frame extraction, grid making and video making functions
def merge_videos(input_video_folder):
    
    os.makedirs("frames/" , exist_ok=True)
  
    input_video_paths = [os.path.join(input_video_folder, f) 
                   for f in os.listdir(input_video_folder) ]

    with concurrent.futures.ThreadPoolExecutor() as executor:

        results = [executor.submit(frame_capture, video, i+1 ) for  i, video in enumerate(input_video_paths)]

        for f in concurrent.futures.as_completed(results):
            print(f.result())

    frames_folder_path = 'frames'    
    frames_paths = [os.path.join(frames_folder_path, f) 
                   for f in os.listdir(frames_folder_path) ]


    #inserting all frames in a list
    all_frames = []
    max_frames = 0
    for frames_folder in frames_paths:
        folder = [os.path.join(frames_folder, f) 
                   for f in os.listdir(frames_folder) if f.endswith('.jpg')]
        folder.sort(key=lambda f: int(re.sub('\D', '', f)))
        all_frames.append(folder)

    #creating blank image
    blank_image = np.zeros((1080,1920,3), np.uint8)
    cv2.imwrite("blank_image.jpg", blank_image)

    #maximum frames 
    max_frames = max(len(frames) for frames in all_frames )

    #making grid with frames
    os.makedirs("grid/" , exist_ok=True)
    
    print('Making grid from frames')
    
    for i in range(max_frames):
        imgs= []
        for j in range(4):
            try: 
                img = all_frames[j][i]
            except:
                img = "blank_image.jpg"
            imgs.append(img)

        make_frame_grid(imgs,i)

    pathIn= 'grid/'
    pathOut = 'merged_video.mp4'
    fps = 25

    print('Making video from Grid Images')
    make_video_from_grid_images(pathIn, pathOut, fps)

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--input_videos_folder_path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    
    
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # new code started here

    # merging videos
    merge_videos(args.input_videos_folder_path)

    # tracking poses
    print('Tracking poses')
    # my code ended here


    cap = cv2.VideoCapture('merged_video.mp4')
    fps = None

    assert cap.isOpened(), f'Faild to load video file merged_video.mp4'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename("merged_video.mp4")}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    while (cap.isOpened()):
        pose_results_last = pose_results

        flag, img = cap.read()
        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro,
            fps=fps)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
