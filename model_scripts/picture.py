import os
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enables GPU dynamic memory allocation (COMMENT OUT IF NOT APPLICABLE)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

"""
SELECT IMAGES
"""
# You may add as many image paths to the array as you wish.
IMAGE_PATHS = ['input/testing_images/testing (19).jpg', 'input/Whats-your-take-on-this-truckers-merge-maneuver-caught-on-dash-cam.png']

"""
LOAD MODEL
"""
PATH_TO_SAVED_MODEL = "TensorFlow/workspace/exported-models/export2/saved_model"

# Let user know the model loading has begun
print('Loading vehicle detection model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_function = tf.saved_model.load('TensorFlow/workspace/exported-models/export2/saved_model')

# Let user know the model has been successfully fetched
end_time = time.time()
elapsed_time = end_time - start_time
print('Retrieved in {} seconds'.format(elapsed_time))

"""
LOAD LABELS
"""
PATH_TO_MODEL_LABELS = 'TensorFlow/workspace/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_MODEL_LABELS, use_display_name=True)

"""
DETECT OBJECTS IN IMAGES
"""
for image in IMAGE_PATHS:

    print('Running inference for {}... '.format(image), end='')
    
    # Loads image into numpy array that will be put into tensorflow graph
    # Returns unit8 numpy array with shape (img_height, img_width, 3)
    image_np = np.array(Image.open(image))
    # Converts input into a tensor
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # pass input tensor into the saved model
    detections = detect_function(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # copy image to be displayed
    image_np_with_detections = image_np.copy()

    # puts class labels, probabilties (score) and boxes on image
    # you can adjust the minimum threshold and number of boxes for varying results
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,
          min_score_thresh=.60,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')

# display images
plt.show()

