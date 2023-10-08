import cv2
import os
import albumentations as A


transform = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=480, height=480),
] , bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


images_folder = "./dataset/test/images"
labels_folder = "./dataset/test/labels"


output_images_folder = "./dataset/test/images2"   #same folder 
output_labels_folder = "./dataset/test/labels2"


def load_image(image_path):
    image = cv2.imread(image_path)
    return image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)



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

def save_labels_in_yolo_format(labels, output_label_path,class_labels):
    with open(output_label_path, 'w') as f:
        i=0
        for label in labels:
            print(label)
            x_center = label[0]
            y_center = label[1]
            width = label[2]
            height = label[3]
            
            # Calculate YOLO format: [class_id, x_center, y_center, width, height]
            yolo_line = f"{class_labels[i]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            i+=1
            f.write(yolo_line)
image_files = os.listdir(images_folder)

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

         output_image_filename = f"cropped_{image_file}"
         output_image_path = os.path.join(output_images_folder, output_image_filename)
         save_image(transformed_image, output_image_path)
        
         # Save augmented labels in YOLO format
         output_label_filename = f"cropped_{label_file}"
         output_label_path = os.path.join(output_labels_folder, output_label_filename)
         save_labels_in_yolo_format(transformed_labels, output_label_path, transformed_class_labels)