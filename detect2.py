# Specify model imports
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
import os
import tensorflow as tf

# Disable GPU if necessary
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Create object detector
class TFObjectDetector():
  
  # Constructor
  def __init__(self):
    self.image_path = "input/160820_233_NYC_TimeSquare5_1080p.mp4"
    self.detect_fn = tf.saved_model.load("./TensorFlow/workspace/exported-models/export1" + "/saved_model")
    self.category_index = label_map_util.create_category_index_from_labelmap("./TensorFlow/workspace/annotations/label_map.pbtxt",
                                                                    use_display_name=True)


  
  # Prepare image
  def prepare_image(self, image):
    return tf.convert_to_tensor(
      np.expand_dims(image, 0), dtype=tf.float32
    )

  
  # Perform detection
  def detect(self, image, label_offset = 1):
    # Ensure that we have a detection function
    assert self.detect_fn is not None
    
    # Prepare image and perform prediction
    image = image.copy()
    image_tensor = self.prepare_image(image)
    detections, predictions_dict, shapes = self.detect_fn(image_tensor)

    # Use keypoints if provided
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in detections:
      keypoints = detections['detection_keypoints'][0].numpy()
      keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          self.category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    
    # Return the image
    return image

  
  # Predict image from folder
  def detect_image(self, path, output_path):

    # Load image
    image = cv2.imread(path)

    # Perform object detection and add to output file
    output_file = self.detect(image)
    
    # Write output file to system
    cv2.imwrite(output_path, output_file)
    
    
  # Predict video from folder
  def detect_video(self, path, output_path):
    
    # Read the video
    vidcap = cv2.VideoCapture(path)
    videoWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_read, image = vidcap.read()
    count = 0

    # Set output video writer with codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (videoWidth, videoHeight))
    
    # Iterate over frames and pass each for prediction
    while frame_read:
        
      # Perform object detection and add to output file
      output_file = self.detect(image)
      
      # Write frame with predictions to video
      out.write(output_file)
      
      # Read next frame
      frame_read, image = vidcap.read()
      count += 1
        
    # Release video file when we're ready
    out.release()

  
if __name__ == '__main__':
  detector = TFObjectDetector()
  detector.detect_video('input/160820_233_NYC_TimeSquare5_1080p.mp4', 'output/v1o.mp4')