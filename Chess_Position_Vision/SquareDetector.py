from cv2 import cuda
from torch import device
from ultralytics import YOLO
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageDraw
import numpy as np
import os

class SquareDetector():
    
    def __init__(self, model,  ):
        self.model = model

    def get_corners_cordinates(self, image_path:str, confidance:float=0.5, imgsz:int=640)-> list:
        

        results = self.model.predict("1.jpg", imgsz=640, conf=0.4)
        

        bboxes = []
        for result in results:
            bboxes_formats = result.boxes

            for bbox in bboxes_formats.xyxy:
                x_min, y_min, x_max, y_max = bbox
                center_x, center_y = self.calculate_center(x_min.item(), y_min.item(), x_max.item(), y_max.item() )
                bboxes.append([center_x, center_y])

        return bboxes 
    
    def draw_bboxes_and_points(self, image_path:str, bboxes:list, 
                               box_colour:str="red", box_width:int = 4, point_size:int = 10, point_colour:str="blue"):
        """
        Draw bounding boxes and points on an image.

        Parameters:
        - image: The image to draw on.
        - bboxes: A list of bounding boxes in the format (x, y, w, h).
        - points: A list of points in the format (x, y).
        - color: The color of the boxes and points. Defaults to green.
        - thickness: The thickness of the boxes and points. Defaults to 2.
        """
        
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Draw each bounding box
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            center_x, center_y = self.calculate_center(x_min, y_min, x_max, y_max )


            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_colour, width=box_width)
        
            draw.ellipse([center_x - point_size, center_y - point_size, center_x + point_size, center_y + point_size], fill=point_colour)
          
        #image.show()
        return image


    def calculate_center(self, x_min, y_min, x_max, y_max):
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

    def create_mosaic(images:list):



        height, width, _ = images[0].shape

        # Create a blank mosaic
        mosaic = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)

        # Fill the mosaic with the provided images
        for i in range(2):
            for j in range(2):
                index = i * 2 + j
                if index < len(images):
                    mosaic[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = images[index]

        cv2.imwrite("test_predict.jpg", mosaic)

def main():
    
    TEST_IMAGE_PATH = r"D:\Dokumenti\Python\Diplomska\Scripts\Chess_Position_Vision\Chess_Position_Vision\Anoteded\test"

    sq_detector = SquareDetector(model=YOLO("runs/detect/yolov8n_corners_fixesdataset__epoch_version50") ) 
    
    
    all_images = [f for f in os.listdir(TEST_IMAGE_PATH ) if '.jpg' in f]
        
    images = all_images[:16]
    
    predicted_images = []
    for image_name in images:
        
        image_path = os.path.join(TEST_IMAGE_PATH , image_name)
        
        bboxes = sq_detector.get_corners_cordinates(image_path=image_path )
        predicted_images.append(sq_detector.draw_bboxes_and_points(image_path=image_path, bboxes=bboxes) )
    
    








if __name__ == "__main__":
    main()