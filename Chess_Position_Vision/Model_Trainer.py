import numpy as np
from cv2 import cuda
from torch import device
from ultralytics import YOLO
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageDraw
from DataProcessor import main_augmentor

def get_gpu_memory_info():
    if torch.cuda.is_available():
        # Get the current GPU device
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)

        print(f"GPU Name: {gpu_properties.name}")
        print(f"GPU Total Memory: {gpu_properties.total_memory / (1024 ** 2):.2f} MB")
        print(f"GPU Free Memory: {torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
        print(f"GPU Allocated Memory: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
    else:
        print("No GPU available.")

def train():
    # Load the model.
    model = YOLO('yolov8n.pt')

    for num_of_epochs in range(10,30,5):
        print("___________________________________epoch:", num_of_epochs, "______________________________________")
        try:
            # Training.
            results = model.train(
                data='data_corners.yaml',
                imgsz=640,
                epochs=num_of_epochs,
                batch=4,
                name='yolov8n_corners' +'_fixesdataset_' + '_epoch_' + str(num_of_epochs),
                plots=True,
            )
        except Exception as e:
            print(f"failed to train on: {num_of_epochs} . Error is {e}")

def calculate_center(x_min, y_min, x_max, y_max):
    """
    Calculate the center of a bounding box.

    Parameters (standart yolov8):
    - x_min, y_min: Coordinates of the top-left corner of the bounding box.
    - x_max, y_max: Coordinates of the bottom-right corner of the bounding box.

    Returns:
    - Tuple (center_x, center_y): Coordinates of the center of the bounding box.
    """
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y
    
def draw_bbox_with_center_on_image(image, bboxes, box_colour:str="red", box_width:int = 4, point_size:int = 10, point_colour:str="blue"):
    
    draw = ImageDraw.Draw(image)
    
    # Loop through the detected objects in this result and draw their bounding boxes
    for box in bboxes.xyxy:

        x_min, y_min, x_max, y_max = box
        center_x, center_y = calculate_center(x_min.item(), y_min.item(), x_max.item(), y_max.item() )
        
        #draw
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_colour, width=box_width)
        
        draw.ellipse([center_x - point_size, center_y - point_size, center_x + point_size, center_y + point_size], fill=point_colour)
        
    
    # Display the image with bounding boxes
    image.show()
    
class model_tester():
    def __init__(self, model):
        self.model = model

    def test_on_image(self, image_name:str):
        
        #results = model_trained.predict(source='test_image.jpg', line_thickness=1, conf=0.01, save_txt=False, save=False
        results = self.model.predict(image_name, imgsz=640, conf=0.4)  # return a list of Results objects
        # Load the original image
        image = Image.open(image_name)
    

        # Iterate through the list of results and draw bounding boxes on the image
        for result in results:
            bounding_boxes = result.boxes  # Get the bounding boxes
            draw_bbox_with_center_on_image(image=image, bboxes=bounding_boxes)       


def test():


    image_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test7.jpg']
    
    for image in image_names:
        

    
        model1 = model_tester(YOLO("runs/detect/yolov8n_corners_fixesdataset__epoch_1010/weights/best.pt"))
        model1.test_on_image(image)

        
    pass








if __name__ == "__main__":

    train()
