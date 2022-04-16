import os
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import matplotlib
import cv2
matplotlib.use('TkAgg')

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enables GPU dynamic memory allocation (COMMENT OUT IF NOT APPLICABLE)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
print('Retreived in {} seconds'.format(elapsed_time))

"""
LOAD LABELS
"""
PATH_TO_MODEL_LABELS = 'TensorFlow/workspace/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_MODEL_LABELS, use_display_name=True)

"""
CONNECT TO WEBCAM
"""
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
# Get width and height of resizing
videoWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

"""
DETECT IN VIDEO
"""
# Loop through each frame of input video and detect
while True:
    # Read frame from video
    image_np = cap.read()
    image_np = np.array(image_np)

    # Convert frame to tensor object
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Pass frame into model
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

    # Display output
    cv2.imshow('Vehicle Detection', cv2.resize(image_np_with_detections, (videoWidth, videoHeight)))

    # Exit video by pressing 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()