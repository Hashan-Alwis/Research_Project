import cv2
import os
from mtcnn.mtcnn import MTCNN

def face_detector(image_path, new_img_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read the image {image_path}")
            return None
        
        detector = MTCNN()
        output = detector.detect_faces(img)
        
        if output:
            x, y, w, h = output[0]["box"]
            height, width, _ = img.shape
            x1 = max(0, x - 20)
            y1 = max(0, y - 20)
            x2 = min(width, x + w + 20)
            y2 = min(height, y + h + 20)
            only_face = img[y1:y2, x1:x2]
            
            os.makedirs(new_img_path, exist_ok=True)
            image_name = os.path.basename(image_path)
            output_path = os.path.join(new_img_path, image_name)
            cv2.imwrite(output_path, only_face)
            print(f"Saved cropped face to: {output_path}")
            return only_face
        else:
            print(f"No face detected in the image: {image_path}")
            return None
    except Exception as e:
        print(f"Error occurred while processing {image_path}: {e}")
        return None

def get_image_names(folder_path):
    try:
        return [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))
        ]
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return []

def main():
    input_folder_path = input("Enter the folder path where the images are: ").strip()
    output_folder_path = input("Enter the folder path where you want to save the new images: ").strip()

    if not os.path.isdir(input_folder_path):
        print(f"Error: The input folder does not exist: {input_folder_path}")
        return

    if not output_folder_path:
        print("Error: Output folder path cannot be empty.")
        return

    image_list = get_image_names(input_folder_path)

    if not image_list:
        print("No valid images to process.")
        return

    for image_path in image_list:
        face_detector(image_path, output_folder_path)

if __name__ == "__main__":
    main()
