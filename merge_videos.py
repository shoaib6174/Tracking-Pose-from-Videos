import os
import cv2
import re
import numpy as np
from os.path import isfile, join
import concurrent.futures


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
    
def frame_capture(path,video_no):
    print("Extracting frames of video no-%d\n" % video_no)
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
    
def main():
    

    os.makedirs("frames/" , exist_ok=True)
    video_folder = 'input_videos'    
    video_paths = [os.path.join(video_folder, f) 
                   for f in os.listdir(video_folder) ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        secs = [5,4,3,2,1]
        results = [executor.submit(frame_capture, video, i+1 ) for  i, video in enumerate(video_paths)]

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

if __name__ == '__main__':
    main()
