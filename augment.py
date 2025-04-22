import os
import cv2
import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(filename='image_augmentation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ImageAugmentor:
    """
    A class for augmenting images with various transformations such as rotation, flipping,
    brightness adjustment, cropping, resizing, grayscale conversion, and more.
    
    Attributes:
        image_folder (str): Path to the folder containing images.
        max_augmentations (int): Maximum number of augmented images per original image.
        valid_extensions (tuple): Valid image file extensions.
        processed_ids (list): List of processed image IDs.
    """
    def __init__(self, image_folder, max_augmentations=15):
        """
        Initializes the ImageAugmentor with the given image folder and augmentation limit.
        
        Args:
            image_folder (str): Path to the image directory.
            max_augmentations (int, optional): Maximum number of augmentations. Default is 15.
        """
        self.image_folder = image_folder
        self.valid_extensions = (".jpg", ".jpeg", ".png")
        self.processed_ids = []
        self.max_augmentations = max_augmentations
        logging.info("ImageAugmentor initialized with folder: %s", image_folder)

    def augment_image(self, img):
        """
        Applies various augmentation techniques to an input image.
        
        Args:
            img (numpy.ndarray): Input image in BGR format.
        
        Returns:
            list: A list of augmented images (numpy.ndarrays), up to max_augmentations.
        """
        augmented_images = []
        
        try:
            # Original Image
            augmented_images.append(img)

            h, w = img.shape[:2]

            # Rotation
            angle = random.choice([-15, 15])
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented_images.append(rotated)

            # Horizontal Flip
            flipped = cv2.flip(img, 1)
            augmented_images.append(flipped)

            # Brightness Adjustment
            brightness_factor = random.uniform(0.7, 1.3)
            bright_img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
            augmented_images.append(bright_img)

            # Cropping (center crop)
            crop_size = int(0.9 * min(h, w))
            x_start = (w - crop_size) // 2
            y_start = (h - crop_size) // 2
            cropped = img[y_start:y_start+crop_size, x_start:x_start+crop_size]
            cropped = cv2.resize(cropped, (w, h))
            augmented_images.append(cropped)

            # Resize to 128x128
            resized = cv2.resize(img, (128, 128))
            augmented_images.append(resized)

            # Convert to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            augmented_images.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

            # Convert to RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented_images.append(rgb)

            # Random Shear
            shear_factor = random.uniform(-0.2, 0.2)
            shear_matrix = np.array([[1, shear_factor, 0], [shear_factor, 1, 0]], dtype=np.float32)
            sheared = cv2.warpAffine(img, shear_matrix, (w, h))
            augmented_images.append(sheared)

            # Blur
            blur_ksize = random.choice([3, 5])
            blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
            augmented_images.append(blurred)

            # Exposure Adjustment (Gamma Correction)
            gamma = random.uniform(0.5, 1.5)
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
            exposure_adjusted = cv2.LUT(img, table)
            augmented_images.append(exposure_adjusted)

            # Random Noise
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, noise)
            augmented_images.append(noisy_img)

        except Exception as e:
            logging.error("Error during augmentation: %s", e)
        
        return augmented_images[:self.max_augmentations]

    def process_images(self):
        """
        Processes images in the specified folder, applies augmentations, and saves them.
        """
        try:
            for foldername in sorted(os.listdir(self.image_folder)):
                face_folder = os.path.join(self.image_folder, foldername, "face")
                
                if os.path.isdir(face_folder):
                    augment_folder = os.path.join(self.image_folder, foldername, "augment")
                    os.makedirs(augment_folder, exist_ok=True)

                    for filename in sorted(os.listdir(face_folder)):
                        if filename.lower().endswith(self.valid_extensions):
                            try:
                                image_path = os.path.join(face_folder, filename)
                                img = cv2.imread(image_path)

                                if img is None:
                                    logging.warning("Skipping unreadable file: %s", filename)
                                    continue

                                id_name = os.path.splitext(filename)[0]
                                self.processed_ids.append(id_name)
                                
                                augmented_imgs = self.augment_image(img)

                                for idx, aug_img in enumerate(augmented_imgs):
                                    aug_path = os.path.join(augment_folder, f"{id_name}_aug{idx}.jpg")
                                    cv2.imwrite(aug_path, aug_img)
                                    logging.info("Saved augmented image: %s", aug_path)
                            
                            except Exception as e:
                                logging.error("Error processing file %s: %s", filename, e)
        
            logging.info("Augmentation completed! Images saved in the 'augment' folder.")
            logging.info("Total images processed: %d", len(self.processed_ids))
            logging.info("Processed IDs: %s", self.processed_ids)
        
        except Exception as e:
            logging.critical("Unexpected error: %s", e)

if __name__ == "__main__":
    """
    Main execution: Initializes the ImageAugmentor and processes images.
    """
    image_folder = "image"
    augmentor = ImageAugmentor(image_folder, max_augmentations=15)
    augmentor.process_images()
