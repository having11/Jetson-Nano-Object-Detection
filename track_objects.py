#!/usr/bin/env python3

from jetbot import ObjectDetector
import sys

model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

from jetbot import Camera

camera = Camera.instance(width=300, height=300)

detections = model(camera.value)

print(detections)

if len(detections) > 0:
    image_number = 0
    object_number = 0

    #print(detections[image_number][object_number])

import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np

""" use:
sudo apt-get install mercurial
hg clone https://bitbucket.org/pygame/pygame
cd pygame
sudo apt-get install python3-dev python3-numpy libsdl-dev libsdl-image1.2-dev \
  libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev \
  libavformat-dev libswscale-dev libjpeg-dev libfreetype6-dev
sudo apt-get install python3-setuptools
python3 setup.py build
sudo python3 setup.py install
"""

import pygame
from pygame.locals import *
from pygame import *

device = torch.device('cuda')

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

from jetbot import bgr8_to_jpeg

width = int(300)
height = int(300)

def detection_center(detection):
    """Computes the center x, y coordinates of the object"""
    #print(detection)
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)
    
def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections):
    """Finds the detection closest to the image center"""
    closest_detection = None
    #print(detections)
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection
        
pygame.init()
screen = pygame.display.set_mode((width*2, height*2), FULLSCREEN | 
    HWSURFACE | DOUBLEBUF)
myfont = pygame.font.SysFont('Verdana', 20)

def execute(change):
    image = change['new']
    detections = model(image)
    #print(detections[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # draw all detections on image
    for det in detections[0]:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
    frame = pygame.surfarray.make_surface(image)
    frame = pygame.transform.smoothscale(frame, (width*2, height*2))
    for det in detections[0]:
        bbox = det['bbox']
        text = myfont.render(str(det['label']), True, (255, 255, 255))
        text = pygame.transform.flip(text, True, False)
        text = pygame.transform.rotate(text, 270)
        frame.blit(text, (int(width * 2 * bbox[0]), int(height * 2 * bbox[1])))
    #matching_detections = [d for d in detections[0]]
    
    # get detection closest to center of field of view and draw it
    """det = None
    if matching_detections != None:
        det = closest_detection(matching_detections)
    if det is not None:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)
        center = detection_center(det)
    """
    #pygame.surfarray.blit_array(screen, image)
    
    frame = pygame.transform.flip(frame, True, False)
    frame = pygame.transform.rotate(frame, 270)
    screen.blit(frame, (0,0))
    pygame.display.update()

    #image_widget.value = bgr8_to_jpeg(image)

    for event in pygame.event.get():
        if event.type == KEYDOWN:
            pygame.quit()
            camera.unobserve_all()
            sys.exit()
execute({'new': camera.value})

camera.unobserve_all()
camera.observe(execute, names='value')