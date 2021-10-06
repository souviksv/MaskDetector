#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import datasets, models, transforms
from PIL import Image
import cv2


class MaskDetector:
    def __init__(self):
        
        device = torch.device("cpu")
        filepath = 'models/mask1_model_resnet101.pth'
        self.model = torch.load(filepath)
        self.model.eval()
        self.model.cpu()

        self.class_names = ['with_mask',
        'without_mask'
        ]

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # TODO: Process a PIL image for use in a PyTorch model
        pil_image = image
    
        image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = image_transforms(pil_image)
        return img
        
    def classify_mask(self, image):
        if image is not None:
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(image)
            image = self.process_image(im)
            img = image.unsqueeze_(0)
            img = image.float()
            output = self.model(image)
            _, predicted = torch.max(output, 1)

            classification1 = predicted.data[0]
            index = int(classification1)
            return self.class_names[index]










