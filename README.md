# YOLO-NAS + Meta segment anything(SAM) object segmentation Application
### About 

combine YOLO-NAS and SAM for object segmentation.The YOLO-NAS model is used for the object detection. The bounding box obtained from the object detection model is passed as a prompt to the SAM model with the returns a segmentation masks of the specfied filter class. 
 
### install the requirements
```
pip install -r requirements.txt
```

### install Segment-Anything Model
```
!pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```
### Download pre-trained SAM model into the root directory. 
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
### Run Demo

-  run the following command:
```
python main.py --video_path "[your video path]" 
```
for example 
```
python main.py --video_path sample2.mp4 --filter classes 'car'
```
-  see the result directory. 
