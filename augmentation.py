import cv2
import os
import albumentations as A
import time

# Define the augmentation transformations
"""

transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels']))

used for first training 
"""

transform = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=480, height=480),
   A.HorizontalFlip(p=0.05),
   A.VerticalFlip(p=0.05),
    A.RandomBrightnessContrast(p=0.2),
    #A.ColorJitter(p=0.05),
    A.Rotate(limit=180, p=0.5),  # Random rotation up to 180 degrees
    A.RandomScale(scale_limit=0.2, p=0.5),  # Random scaling up to 20%
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Paths to your images and labels folders
images_folder = "./dataset/images"
labels_folder = "./dataset/labels"

# Output folder for augmented images and labels
output_images_folder = "./dataset/train/images"
output_labels_folder = "./dataset/train/labels"

# Load and save image using OpenCV
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# Load and save labels (implement as needed)
"""
def load_labels(label_path):
    # Implement the logic to load YOLO-format labels using OpenCV
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

    # Convert YOLO coordinates to absolute values
    x_min = (x_center - width / 2) * width
    y_min = (y_center - height / 2) * height
    x_max = (x_center + width / 2) * width
    y_max = (y_center + height / 2) * height

    # Calculate Albumentations format: [x_min, y_min, x_max, y_max, class_name]
    albumentations_bbox = [x_min, y_min, x_max, y_max, class_id]
   # labels = ...  # Load labels using OpenCV
    return albumentations_bbox


"""

def load_labels(label_path):
    labels = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            label = [x_center, y_center, width, height]
            labels.append(label)

    return labels

def load_class(label_path):
    classes = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            classes.append(class_id)

    return classes

""" 
def save_labels_in_yolo_format(labels, output_label_path):
  # Implement the logic to save labels in YOLO format using OpenCV
    ...  # Save labels in YOLO format using OpenCV

    # YOLO-format: [x_center, y_center, width, height]
    
   

"""
def save_labels_in_yolo_format(labels, output_label_path,class_labels):
    with open(output_label_path, 'w') as f:
        i=0
        for label in labels:
            print(label)
            x_center = float(label[0])
            y_center = float(label[1])
            width = float(label[2])
            height = float(label[3])
            
            # Calculate YOLO format: [class_id, x_center, y_center, width, height]
            yolo_line = f"{class_labels[i]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            i+=1
            f.write(yolo_line)
  
# Iterate through images and labels


image_files = os.listdir(images_folder)


for _ in range(2):
    for image_file in image_files:
        
         image_path = os.path.join(images_folder, image_file)
    
        # Load corresponding label file
         label_file = os.path.splitext(image_file)[0] + ".txt"
         label_path = os.path.join(labels_folder, label_file)
        
         # Load image and labels
         image = load_image(image_path)
         labels = load_labels(label_path)

        
        # Apply transformations
         class_labels=load_class(label_path)
        # Apply transformations
         transformed = transform(image=image, bboxes=labels, class_labels=class_labels)
         transformed_image = transformed['image']
         transformed_labels = transformed['bboxes']
         transformed_class_labels=transformed["class_labels"]

        
        # Generate unique filenames using a timestamp
         timestamp = int(time.time())
         output_image_filename = f"augmented_{timestamp}_{image_file}"
         output_image_path = os.path.join(output_images_folder, output_image_filename)
         save_image(transformed_image, output_image_path)
        
         # Save augmented labels in YOLO format
         output_label_filename = f"augmented_{timestamp}_{label_file}"
         output_label_path = os.path.join(output_labels_folder, output_label_filename)
         save_labels_in_yolo_format(transformed_labels, output_label_path, transformed_class_labels)

    