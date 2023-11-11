from cv2 import cuda
from torch import device
from ultralytics import YOLO
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image, ImageDraw


class SquareDetector():
    
    def __init__(self, model,  ):
        self.model = model

        
    def get_corners_cordinates(self, image_path:str, confidance:float=0.5, imgsz:int=640)-> list:
        results = self.model.predict(image_path, imgsz=640, conf=0.5)
        
        bboxes = []
        for result in results:
            boxes_formats = result.boxes

            for box in boxes_formats.xyxy: #there are difrent formats you can extract
                bboxes.append(box)

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
            center_x, center_y = self.calculate_center(x_min.item(), y_min.item(), x_max.item(), y_max.item() )


            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_colour, width=box_width)
        
            draw.ellipse([center_x - point_size, center_y - point_size, center_x + point_size, center_y + point_size], fill=point_colour)
          
        image.show()


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


def main():
    
    sq_detector = SquareDetector(model=YOLO("runs/detect/yolov8n_corners8/weights/best.pt"))
    
    bboxes = sq_detector.get_corners_cordinates(image_path='test2.jpg')
    sq_detector.draw_bboxes_and_points(image_path='test2.jpg', bboxes=bboxes)





if __name__ == "__main__":
    main()