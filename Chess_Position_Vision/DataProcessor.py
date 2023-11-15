from binascii import a2b_uu
import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import random
import shutil
from sklearn.model_selection import train_test_split

"""
INPUT_DIR -> Should be the zipped folder dowloaded from roboflow.
Structure should look like:
-INPUT_DIR
        -train
        -test
        -valid
        -other files(ignored)
            
OUTPUT_DIR -> meant to store the augmentated images. If need be, agumentations can be done on the previus agumentations
    
EXPORT_DIR -> A directory, where the train, test and valid images/bboxes will be combined. Becouse (recommended) we...
...only augment train images, when using export direcotry where the valid and test images/bboxes are is needed (just INPUT_DIR)
"""

class DataAugmentor:
    """
    Class for augmentating images and associated bounting boxes (bboxes).
    Each augmentation is done separately and image&bbox are saved each time, meaning all combinations of augmentations can be done... 
    ... If you choose output_dir for the input directory, augmentations will be performed on alredy augmentated images, creating combinations of augmentations (creating many many more photos)
    """
    def __init__(self, num_of_rotations =20, img_width = 640, img_hight=640):
        self.num_of_rotations = num_of_rotations 
        self.img_width = img_width
        self.img_high = img_hight

    def __convert_bbox_from_yolo_to_cv2(self,bbox):
        """
        Converts values to int, mapps, calculates the diffrent dimension of a bbox by other defenition (cv2)
        """
        # Unpack the bounding box coordinates
        center_x, center_y, w, h, label = bbox


        # Convert to top-left x, y
        x = int((center_x - w / 2) * 640)
        y = int((center_y - h / 2) * 640)

        # Convert width and height from [0, 1] to pixel values
        w = int(w * 640)
        h = int(h * 640)
        
        return x,y,w,h
    
    def __delete_contents_of_folder(self, folder_path):
        # Check if the path exists
        if os.path.exists(folder_path):
            # Iterate over all the files and subdirectories in the folder
            for file_or_dir in os.listdir(folder_path):
                file_or_dir_path = os.path.join(folder_path, file_or_dir)
            
                # Check if it is a file
                if os.path.isfile(file_or_dir_path):
                    # Delete the file
                    os.remove(file_or_dir_path)
                # Check if it is a subdirectory
                elif os.path.isdir(file_or_dir_path):
                    # Use shutil.rmtree to delete the subdirectory and its contents recursively
                    shutil.rmtree(file_or_dir_path)
    
        else:
            print(f"The path {folder_path} does not exist.")

    def __get_image_file_names(self,images_path:str)->list:
        """
        Returns array/list of file names. If .jpg is somehere in the name 
        """

        return  [f for f in os.listdir(images_path) if '.jpg' in f]
    
    def __get_bboxes(self, labels_path:str, img_name:str)->list:
        """
        Opens label txt document with the same name as the picture (it replaces .jpg with .txt)
        """
        bboxes=[]
        with open( os.path.join(labels_path, img_name.replace('.jpg', '.txt') ) ) as labels:
            for line in labels:
                # Convert string to float 
                bbox = [float(num) for num in line.strip().split()]
                    
                #class label should be at the end. If at the start, move it back
                if bbox[0].is_integer():
                    bbox.append(bbox.pop(0))
                    
                bboxes.append(bbox)
            return bboxes
        
    def __save_augmentated_img_and_bboxes_to_new_dir(self, output_dir_path:str, image_file_name:str, image, bboxes:list, rotation_index:int = 0):
        """
        Saves the image and the box with the same name + the rotation index to the output directory
        Structure should be like this:
        - output_dir_path
            - images
            - labels
        It also shifts class name back to the 0 (it was shifted in bbox augmentations; it had to be at te back)
        """
        # Create full path for the image and bounding box files
        image_path = os.path.join(output_dir_path,  'rotation' + str(int(rotation_index)) + image_file_name  )
        bbox_path = os.path.join(output_dir_path, 'rotation'+ str(int(rotation_index)) + image_file_name.replace('.jpg', '.txt')  )

        # Save the image
        cv2.imwrite(image_path, image)

        # Save the bounding boxes
        with open(bbox_path, 'w') as f:
            for bbox in bboxes:
                # Convert tuple to list
                bbox = list(bbox)
                # Fix class position
                if not bbox[0].is_integer():
                    bbox.insert(0, bbox.pop())
                
                # Write each bounding box to the file
                f.write(' '.join(map(str, bbox)) + '\n')
                
    def __copy_files_to_directory(self, files, input_dir:str, output_dir:str):
        """
        Saves any file to outputimage or bboxes
        """
        #We put None to test_img and test_txt sometimes
        if files is None:
            return
        
        for file in files:
            src_path = os.path.join(input_dir, file)
            dest_path = os.path.join(output_dir, file)
            shutil.copy(src_path, dest_path)
                
    def copy_original_img_and_bboxes(self, input_dir, output_dir):
        """
        Copies images and bboxes to output directory.
        Standart directory structure must be created beforehand for both output and input:
        - output_dir_path
            - images
            - labels
        """

        #get all image file names
        image_file_names = self.__get_image_file_names(input_dir )

        #Do stuff on all images/bboxes
        for img_name in image_file_names:
            # load the image with file name 
            image = cv2.imread(os.path.join(input_dir, img_name))
            bboxes = self.__get_bboxes(input_dir, img_name.replace('.jpg', '.txt') )
            
            self.__save_augmentated_img_and_bboxes_to_new_dir(output_dir, img_name, image=image, bboxes=bboxes )
            
        print(f"Copied original images from {input_dir} to {output_dir}")
  
    def rotate_imgbboxes(self, num_of_rotations:int, input_dir, output_dir):
        """
        Generates many more images with bboxes. 
        Saves them to new directory 
        """
        #get all image file names
        image_file_names = self.__get_image_file_names(input_dir)

        #Do stuff on all images/bboxes
        for img_name in image_file_names:
            
            # load the image with file name 
            image = cv2.imread(os.path.join(input_dir, img_name))
            bboxes = self.__get_bboxes(input_dir, img_name.replace('.jpg', '.txt') )

            # Apply rotations to the original image 
            for rotation_index in range(num_of_rotations):
                angle = random.uniform(5, 360)
                transform = A.Compose(
                    #safe_rotate resizes and rotates->avoids black corners if possible. Use A.Rotate to rotate round center
                    [A.SafeRotate(limit=(angle, angle), border_mode=cv2.BORDER_CONSTANT, value=0)],
                    bbox_params=A.BboxParams(format='yolo')
                )

                transformed = transform(image=image, bboxes=bboxes)
 
                #save img and bboxes to new dir (output_dir)
                self.__save_augmentated_img_and_bboxes_to_new_dir(output_dir, img_name, image=transformed['image'], bboxes=transformed['bboxes'], rotation_index=rotation_index )
                
        print(f"Rotated images/bboxes from {input_dir} to {output_dir}")
                
    def split_roboflow_dataset(self, input_dir:str, export_dir:str, train_size:float=0.8, val_size:float=0.1, test_size:float=0.1)-> None:
        
        #robboflow splits images and bboxes
        input_img_dir = os.path.join(input_dir, "images")
        input_bboxes_dir = os.path.join(input_dir, "labels")
        

        if train_size + val_size + test_size != 1.0:
            raise Exception("You are an idiote")
        
        #get names of images and bboxes
        image_file_names = self.__get_image_file_names(input_img_dir )
        labels_file_names = [img_name.replace('.jpg', '.txt') for img_name in image_file_names]
   
        # Split the data into train, test, and validation sets
        image_train, image_temp, text_train, text_temp = train_test_split(image_file_names, labels_file_names, test_size= (1.0 - train_size) , random_state=42)
        
        #if we choose to futher split to val and test 
        if test_size:
            image_test, image_val, text_test, text_val = train_test_split(image_temp, text_temp, test_size= ( test_size/(val_size+test_size) ) , random_state=42)
        else:#put all the temp to the evalude, and None to test so it's not saved
            image_val = image_temp
            text_val = text_temp
            image_test = None
            text_test = None
            
        
        self.__copy_files_to_directory(image_train, input_dir=input_img_dir, output_dir= os.path.join(export_dir, "train")  )
        self.__copy_files_to_directory(image_test, input_dir=input_img_dir, output_dir= os.path.join(export_dir, "test") )
        self.__copy_files_to_directory(image_val, input_dir=input_img_dir, output_dir= os.path.join(export_dir, "valid") )
        
        self.__copy_files_to_directory(text_train, input_dir=input_bboxes_dir, output_dir= os.path.join(export_dir, "train") )
        self.__copy_files_to_directory(text_test, input_dir=input_bboxes_dir, output_dir= os.path.join(export_dir, "test") )
        self.__copy_files_to_directory(text_val, input_dir=input_bboxes_dir, output_dir= os.path.join(export_dir, "valid") )
        
        print(f"Exported images/bboxes from {input_dir} to {export_dir}")
        
    def apply_grayscale_to_all(self, input_dir: str):
        #get all image file names
        image_file_names = self.__get_image_file_names(os.path.join( input_dir, "images") )

        #Do stuff on all images/bboxes
        for img_name in image_file_names:
            # load the image with file name 
            image = cv2.imread(os.path.join(input_dir, img_name))

            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image with the same filename
            output_path = os.path.join(input_dir,"images", img_name)
            cv2.imwrite(output_path, grayscale_image)
                
        print(f"Applied gray scale to images from {input_dir}")
        
    def flip_imgbboxes(self, axis:str, input_dir, output_dir):
        """
        Generates horizontally flipped images with bboxes and saves them to new directory  
        axis:
            - 'horizontal'
            - 'vertical'
        """

        augmentations = {
            'horizontal': A.HorizontalFlip(p=1),
            'vertical': A.VerticalFlip(p=1)
        }

        #get all image file names
        image_file_names = self.__get_image_file_names(input_dir)

        #Do stuff on all images/bboxes
        for img_name in image_file_names:
        
            # load the image with file name 
            image = cv2.imread(os.path.join(input_dir, img_name))
            bboxes = self.__get_bboxes(input_dir, img_name.replace('.jpg', '.txt') )

            # Apply the chosen flip to the original image 
            transform = A.Compose(
                    [augmentations[axis]],
                    bbox_params=A.BboxParams(format='yolo')
                )


            # apply the selected flip to the original image 
            transformed = transform(image=image, bboxes=bboxes)

            #save img and bboxes to new dir (output_dir)
            self.__save_augmentated_img_and_bboxes_to_new_dir(output_dir, img_name, image=transformed['image'], bboxes=transformed['bboxes'])
        
        print(f"Flipped images/bboxes from {input_dir} to {output_dir}")
        
    def export_augmented_to_dataset(self, input_dir:str, output_dir:str):
        """
        Adss images back the datset
        """
        
        image_file_names = self.__get_image_file_names(input_dir)
        labels_file_names = [img_name.replace('.jpg', '.txt') for img_name in image_file_names]

        self.__copy_files_to_directory(image_file_names, input_dir, output_dir)
        self.__copy_files_to_directory(labels_file_names, input_dir, output_dir)
        
    def display_image_bboxes(self,image, bboxes):
        # Copy the image to avoid changing the original
        img_copy = np.copy(image)

        # Loop over all bounding boxes and draw them on the image
        for bbox in bboxes:

            x,y,w,h = self.__convert_bbox_from_yolo_to_cv2(bbox)
            # Draw the bounding box on the image
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the image
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.show()      
        

    
def main_augmentor():

    ROBOFLOW_EXPORT_DIR = r"D:\Prenosi\Corner_Detection_Roboflow\data"
    DATASET_DIR = r"D:\Dokumenti\Python\Diplomska\Dataset\Corners\Anoteded"
    EXPORT_DIR = os.path.join(DATASET_DIR, "train")
    TEMPORARY_FOLDER =   r"D:\Dokumenti\Python\Diplomska\Dataset\Corners\Augmeneted"  


    
    augmentor = DataAugmentor()
    
    augmentor.split_roboflow_dataset(input_dir=roboflow_export_dir, export_dir=dataset_dir )
    augmentor.copy_original_img_and_bboxes(input_dir=export_dir , output_dir=temporary_folder)
    augmentor.flip_imgbboxes(axis='horizontal', input_dir=temporary_folder, output_dir=temporary_folder)
    augmentor.flip_imgbboxes(axis='vertical', input_dir=temporary_folder, output_dir=temporary_folder)
    augmentor.rotate_imgbboxes(num_of_rotations=2, input_dir=temporary_folder, output_dir=temporary_folder)
    

    augmentor.export_augmented_to_dataset(input_dir=TEMPORARY_FOLDER, output_dir=EXPORT_DIR)



if __name__ == "__main__":
    main_augmentor()