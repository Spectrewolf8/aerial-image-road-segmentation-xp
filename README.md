# Model Card for spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp

This model card provides an overview of a computer vision model designed for aerial image road segmentation using the U-Net-50 architecture. The model is intended to accurately identify and segment road networks from aerial imagery, crucial for applications in mapping and autonomous driving.

## Model Details

### Model Description

- **Developed by:**  [spectrewolf8](https://github.com/Spectrewolf8)
- **Model type:** Computer-Vision/Semantic-segmentation
- **License:** MIT

### Model Sources

- **Repository:** https://github.com/Spectrewolf8/aerial-image-road-segmentation-xp
  
## Uses

### Direct Use

This model can be used to segment road networks from aerial images without additional fine-tuning. It is applicable in scenarios where detailed and accurate road mapping is required.

### Downstream Use 

When fine-tuned on additional datasets, this model can be adapted for other types of semantic segmentation tasks, potentially enhancing applications in various remote sensing domains.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
# Import necessary classes
from tensorflow.keras.models import load_model
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
seed=24
batch_size= 8
# Load images for dataset generators from respective dataset libraries. The images and masks are returned as NumPy arrays
# Images can be further resized by adding target_size=(150, 150) with any size for your network to flow_from_directory parameters
# Our images are already cropped to 256x256 so traget_size parameter can be ignored
def image_and_mask_generator(image_dir, label_dir):
    img_data_gen_args = dict(rescale = 1/255.)
    mask_data_gen_args = dict()
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_generator = image_data_generator.flow_from_directory(image_dir, 
                                                               seed=seed, 
                                                               batch_size=batch_size,
                                                               classes = ["."],
                                                               class_mode=None #Very important to set this otherwise it returns multiple numpy arrays thinking class mode is binary.
                                                               )  
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_generator = mask_data_generator.flow_from_directory(label_dir, 
                                                             classes = ["."],
                                                             seed=seed, 
                                                             batch_size=batch_size,
                                                             color_mode = 'grayscale', #Read masks in grayscale
                                                             class_mode=None
                                                             )
    # print processed image paths for vanity
    print(image_generator.filenames[0:5])
    print(mask_generator.filenames[0:5])
    
    generator = zip(image_generator, mask_generator)
    return generator
# Method to calculate Intersection over Union Accuracy Coefficient
def iou_coef(y_true, y_pred, smooth=1e-6):
    intersection = tensorflow.reduce_sum(y_true * y_pred)
    union = tensorflow.reduce_sum(y_true) + tensorflow.reduce_sum(y_pred) - intersection
    
    return (intersection + smooth) / (union + smooth)
# Method to calculate Dice Accuracy Coefficient
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = tensorflow.reduce_sum(y_true * y_pred)
    total = tensorflow.reduce_sum(y_true) + tensorflow.reduce_sum(y_pred)
    
    return (2. * intersection + smooth) / (total + smooth)
# Method to calculate Dice Loss
def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
# Method to create generator
def create_generator(zipped):
    for (img, mask) in zipped:
        yield (img, mask)
model_path = "path"
u_net_model = load_model(model_path, custom_objects={'soft_dice_loss': soft_dice_loss, 'dice_coef': dice_coef, "iou_coef": iou_coef})
test_generator = create_generator(image_and_mask_generator(output_test_image_dir,output_test_label_dir))
# Assuming create_generator is defined and provides images for prediction
images, ground_truth_masks = next(test_generator)
# Make predictions
predictions = u_net_model.predict(images)
# Apply threshold to predictions
thresh_val = 0.8
prediction_threshold = (predictions > thresh_val).astype(np.uint8)
# Visualize results
num_samples = min(10, len(images))  # Use at most 10 samples or the total number of images available
f = plt.figure(figsize=(15, 25))
for i in range(num_samples):
    ix = random.randint(0, len(images) - 1)  # Ensure ix is within range
    f.add_subplot(num_samples, 4, i * 4 + 1)
    plt.imshow(images[ix])
    plt.title("Image")
    plt.axis('off')
    f.add_subplot(num_samples, 4, i * 4 + 2)
    plt.imshow(np.squeeze(ground_truth_masks[ix]))
    plt.title("Ground Truth")
    plt.axis('off')
    f.add_subplot(num_samples, 4, i * 4 + 3)
    plt.imshow(np.squeeze(predictions[ix]))
    plt.title("Prediction")
    plt.axis('off')
    f.add_subplot(num_samples, 4, i * 4 + 4)
    plt.imshow(np.squeeze(prediction_threshold[ix]))
    plt.title(f"Thresholded at {thresh_val}")
    plt.axis('off')
plt.show()
```


## Training Details

### Training Data

The model was trained on the Massachusetts Roads Dataset, which includes high-resolution aerial images with corresponding road segmentation masks. The images were preprocessed by cropping into 256x256 patches and converting masks to binary format.

### Training Procedure

#### Preprocessing

- Images were cropped into 256x256 patches to manage memory usage and improve training efficiency.
- Masks were binarized to create clear road/non-road classifications.

#### Training Hyperparameters

- **Training regime:** FP32 precision
- **Epochs:** 2
- **Batch Size:** 8
- **Learning Rate:** 0.0001

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated using a separate set of aerial images and their corresponding ground truth masks from the dataset.

#### Metrics

- **Intersection over Union (IoU):** Measures the overlap between predicted and actual road areas.
- **Dice Coefficient:** Evaluates the similarity between predicted and ground truth masks.

### Results

The model achieved 71% accuracy in segmenting road networks from aerial images, with evaluation metrics indicating good performance in distinguishing road features from non-road areas.

#### Summary

The U-Net-50 model effectively segments road networks, demonstrating its potential for practical applications in urban planning and autonomous systems.
## Technical Specifications

### Model Architecture and Objective

- **Architecture:** U-Net-50
- **Objective:** Road segmentation in aerial images

### Compute Infrastructure

#### Software

- **Framework:** TensorFlow 2.x
- **Dependencies:** Keras, OpenCV, tifffile

## Demo
![image](https://github.com/user-attachments/assets/37ad462a-3d6c-41ec-ad00-613534b30f4a)

