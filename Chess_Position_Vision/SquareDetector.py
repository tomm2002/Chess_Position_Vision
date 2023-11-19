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
    
    def __init__(self, model ):
        self.model = model

    def predict_bboxes(self, image_path:str, confidance:float=0.5, imgsz:int=640)-> list:
        """
        Returns list of bboxes
        """

        results = self.model.predict(image_path, imgsz=imgsz, conf=confidance)
        
        bboxes = []
        for result in results:
            bboxes_formats = result.boxes

            for bbox in bboxes_formats.xyxy:
                x_min, y_min, x_max, y_max = bbox
                
                bboxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
        return bboxes

    def get_center_points_from_bboxes(self, bboxes):
        center_points = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            center_x, center_y = self.calculate_center(x_min, y_min, x_max, y_max )
            center_points.append([center_x, center_y])
            
        return center_points

    
    def draw_bboxes_and_points(self, image_path:str, bboxes:list, 
                               box_colour:str="red", box_width:int = 4, point_size:int = 5, point_colour:str="blue", show_img = True):
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
          
        if show_img: image.show()
        
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

    def perspective_transform(self, image_path, src_pts):
        image = cv2.imread('1.jpg')

        # Define source points


        # Define destination points
        height, width = image.shape[:2]

        # Define destination points based on the image size
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')

        # Get the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply the perspective transformation
        warped_image = cv2.warpPerspective(image, matrix, (width, height))

        # Save the output
        cv2.imwrite('output.jpg', warped_image)

    def order_points(self, pts):
        # Initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # The top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Return the ordered coordinates
        return rect


def main():
    
    IMAGE_PATH = "1.jpg"

    sq_detector = SquareDetector(model=YOLO("corner_model/weights/best.pt") ) 
    
    
    bboxes = sq_detector.predict_bboxes(image_path=IMAGE_PATH)
    
    points = sq_detector.get_center_points_from_bboxes(bboxes)
    points = sq_detector.order_points(np.array(points, dtype='float32'))

    sq_detector.perspective_transform(IMAGE_PATH, points)
    
    








if __name__ == "__main__":
    main()