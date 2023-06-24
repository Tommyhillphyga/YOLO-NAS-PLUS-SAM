import numpy as np
import cv2
import math
import torch
import argparse

from detection import ImageDetection

def main (args):
    '''
    combine YOLO-NAS and SAM for object segmentation
    The YOLO-NAS model is used for the object detection. 
    The bounding box obtained from the object detection model 
    is passed as a prompt to the SAM model with the returns a 
    segmentation masks of the specfied filter classs. 
    '''
    detection_class = ImageDetection(args.video_path, args.weights, args.filter_classes, args.conf_thres)

    frame = detection_class.run(args.output_path,args.display, save=True)
























if __name__== "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', default='sample2.mp4', help='Path to input video')
    parser.add_argument('--no_save', default=True, action='store_false',
                        dest='save_result', help='whether or not save results')
    parser.add_argument('--no_display', default=False, action='store_false',
                        dest='display', help='whether or not display results on screen')
    parser.add_argument('--output_path', default='output.avi',  help='Path to output directory')
    parser.add_argument('--filter_classes', default= None, help='Filter class name')
    parser.add_argument('-w', '--weights', default= 'sam_vit_h_4b8939.pth', help='Path of trained weights')
    parser.add_argument('-ct', '--conf_thres', default=0.25, type=float, help='confidence score threshold')

    args = parser.parse_args()

    main(args)