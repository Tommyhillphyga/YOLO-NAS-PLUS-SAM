import numpy as np
import cv2
import math
import torch
import matplotlib.pyplot as plt

### import neccessary utilis
from coco_names import class_names
from utils import draw_ui_box

### import neccessary modules
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
from super_gradients.training import models


class ImageDetection():
    def __init__(self, input_file, sam_checkpoint, filter_class, conf_thres) -> None:

        self.input_file = input_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.filter_class = filter_class
        self.conf_thres = conf_thres
        # video writer
        self.video_writer = None
        self.class_names = class_names
        
        # load sam chekpoint
        self.sam_checkpoint = sam_checkpoint   # segment anything pretrained weight | "sam_vit_h_4b8939.pth'
        self.model_type = 'vit_h'
        self.sam = self.load_sam()
        self.yolo_nas_l = self.load_yolo_nas()

    def load_sam(self):
        '''
          load SAM checkpoint and return SAM model
        '''
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        return sam
    
    def load_yolo_nas(self):
        '''
          load YOLO-NAS checkpoint and return YOLO-NAS model
        '''
        yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")
        return yolo_nas_l
    

    def processvideo(self):
       '''
        Initialize opencv videocapture
       '''
       cap = cv2.VideoCapture(self.input_file)
       return cap

    
    def run (self, output_path, display=False, save = True):
        cap = self.processvideo()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 1
        # if cap.isOpened() is False:
        #   print('Error while trying to read video. Please check path again')

        while cap.isOpened(): 
            success, frame = cap.read()
            if not success:
                break
            else:
                print(f"Frame {frame_count}/{int(length)} Processing")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bboxes_xyxy, label_ids = self.get_yolo_nas_boxes(image_rgb)  # predicted bboxes from yolo_nas
                if len(bboxes_xyxy) == 0:
                    continue
                # input_tensor_box = self.get_input_boxes(bboxes_xyxy, self.filter_class, label_ids, frame)
                masks = self.segment_object(bboxes_xyxy, frame, label_ids, self.filter_class)
                for mask in masks:
                    segmented_mask = self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                    rearranged_mask = np.transpose(segmented_mask[:,:,:3], (2,0,1))
                    frame = self.draw_segmented_mask(rearranged_mask, frame)
                    
                frame_count +=1 
                    #draw bbox
                if display:
                    cv2.imshow('frame', frame)
                if save:
                    if self.video_writer is None:
                        self.video_writer = cv2.VideoWriter(f"{str(output_path)}", cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
                    self.video_writer.write(frame)

        print('Process Complete...') 
        cap.release()
        return frame

    def mask_generator(self):
        '''
          Initialize Sam automatic mask generator parameters
          - > parameters can be modified for more optimized performance. 
        '''
        mask_generator_ = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.96,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

        return mask_generator_
    
    def segment_object(self, bbox, frame, class_ids, filter_class):
        img_rgb = frame.copy()
        mask_predictor = SamPredictor(self.sam)
        mask_predictor.set_image(frame)
        input_boxes = self.get_input_boxes(bbox, filter_class, class_ids, frame)
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(input_boxes, img_rgb.shape[:2])
        masks, _, _ = mask_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
        return masks
    
    def draw_segmented_mask(self, anns, frame):
      img = frame.copy()
      mask_annotator = sv.MaskAnnotator(color=sv.Color.blue())
      detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=anns),
        mask=anns)
      detections = detections[detections.area == np.max(detections.area)]
      segmented_mask = mask_annotator.annotate(scene=frame, detections=detections)
      return segmented_mask
    
    def show_mask(self, mask, ax, random_color=False):
      if random_color:
          color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
      else:
          color = np.array([30/255, 144/255, 255/255, 0.6])
      h, w = mask.shape[-2:]
      mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
      return mask_image
    
    def get_input_boxes(self, bboxes, filter_class, class_ids, frame):   
      # get the bbox of all instances of the filtered classs in the frame.  
      if filter_class is None:
        for _, (box, class_id) in enumerate(zip(bboxes, class_ids)):
            label = self.class_names[class_id]
            draw_ui_box(box, frame, label=label, color=None, line_thickness=2)
        input_tensor_box = torch.tensor(bboxes, device=self.device)
      else:
        input_box = []
        for _, (box, class_id) in enumerate(zip(bboxes, class_ids)):
            label = self.class_names[class_id]
            x1,y1,x2,y2 = box
            x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)
            target_id = self.class_names.index(filter_class)
            draw_ui_box(box, frame, label=label, color=(0,20,224), line_thickness=2)        
            if target_id==class_id:
                input_box.append(box)
        input_tensor_box = torch.tensor(input_box, device=self.device)
      return input_tensor_box
    
    
    def get_yolo_nas_boxes(self, image): 
        conf_threshold = self.conf_thres
        yolo_nas_model = self.yolo_nas_l
        predictions = yolo_nas_model.predict(image, conf=conf_threshold)
        detections = predictions._images_prediction_lst
        detection_details = [item for item in detections]
        #Get detection details
        bboxes_xyxy = detection_details[0].prediction.bboxes_xyxy.astype(int).tolist()
        confidence = detection_details[0].prediction.confidence.tolist()
        label_ids= detection_details[0].prediction.labels.astype(int).tolist()

        return bboxes_xyxy, label_ids



       



       



       
       
    