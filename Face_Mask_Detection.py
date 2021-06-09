# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:50:43 2021

@author: 1636740
"""
import mediapipe as mp
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
import os
import argparse


class FaceDetection():
    
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faces = self.mpFace.FaceDetection(min_detection_confidence = self.min_detection_confidence)
        
    def Landmarks(self, frame, draw = True):
        
        self.results = self.faces.process(frame) 
        
        images = []
        pts = []
        
        if self.results.detections:
            if draw:
                frame = self.DrawFaceLandmarks(frame)

            ht, wt, = frame.shape[0], frame.shape[1]
            for i in self.results.detections:    
                x = min(int(i.location_data.relative_bounding_box.xmin*wt), wt-1)
                y = min(int(i.location_data.relative_bounding_box.ymin*ht), ht-1)
                h = min(int(i.location_data.relative_bounding_box.height*ht), ht-1)
                w = min(int(i.location_data.relative_bounding_box.width*wt), wt-1)
                
                images.append(frame[y:y+h,x:x+w,:])
                pts.append([x, y, h ,w])
            
        return frame, images, pts
        
    def DrawFaceLandmarks(self, frame):
        
        for i in self.results.detections:    
            
            self.mpDraw.draw_detection(frame, i)
        
        return frame 
    
    
def set_cwd():
    
    parser = argparse.ArgumentParser(description = 'path of project directory')
    parser.add_argument("-p", "--path", default='/FaceMaskDetection/', 
                    required = True, type = str, help = 'Give path of project directory')
    args = parser.parse_args()
    os.chdir(args.path)


def main():
    
    global flag
    
    t = 0
    count_frames = 0
    cats = ['with_mask', 'without_mask']
    
    video = cv2.VideoCapture(0)  
    video.set(3, 1280)  # width
    video.set(4, 720)   # height
    
    # frame_width = int(video.get(3)) 
    # frame_height = int(video.get(4)) 
    # vid_fps = int(video.get(5)) 
    # code_of_codec = int(video.get(6))
    # No_of_frames = int(video.get(7))  
    # size = (frame_width, frame_height) 
    
    # result = cv2.VideoWriter('C:/Users/1636740/Desktop/tshr/cv2/Magic_1.avi',  
    #                          cv2.VideoWriter_fourcc(*'DIVX'), 
    #                          10, size) 
    
    faces = FaceDetection(min_detection_confidence=0.7)
    
    model = load_model(r'MobileNetV2.h5')
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frame, Images, points = faces.Landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), draw = False)
        
        for n, image in enumerate(Images): 
            
            x, y, w, h = points[n][0], points[n][1], points[n][2], points[n][3]
                
            # preprocessing
            img = cv2.resize(image, (224, 224))
            img = img/255.0                  # if u have used 255.0 division to normalize images while training
            # img = preprocess_input(img)    # if u have used preprocess_input to normalize images while training 
            img = np.expand_dims(img,axis=0)
            
            pred = model.predict(img)
            prediction = cats[pred.argmax()]
            # probability = round(pred[0][pred.argmax()]*100,2)
            
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.5
            FONT_THICKNESS = 2
            text_width, text_height = cv2.getTextSize(prediction, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            
            if prediction == 'without_mask': 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(frame, (x-1, y-text_height-6+3), (x+text_height+3+110, y), (255, 0, 0), -1)
                cv2.putText(frame, prediction.upper(), (x, y-2), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            elif prediction == 'with_mask':
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x-1, y-text_height-6+3), (x+text_height+3+80, y), (0, 255, 0), -1)
                cv2.putText(frame, prediction.upper(), (x, y-2), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
            
            s = time.time()
            fps = int(1/(s-t))
            t = s
                
            cv2.putText(frame, 'CPU_FPS: '+str(fps), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            
            
            cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
        # result.write(output)
        
        count_frames += 1
        print(count_frames)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
        count_frames += 1
        
    # result.release()    
    video.release() 
    
    cv2.destroyAllWindows()
    print("Done processing video")
    
    return None
    


if __name__ == '__main__':
    # set_cwd()
    main()