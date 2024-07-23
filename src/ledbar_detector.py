#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/caselab/redbot04_ws/src/yolov5/weights/best.pt')
rospy.loginfo("YOLOv5 model loaded successfully.")

# Initialize ROS node
rospy.init_node('yolov5_node', anonymous=True)
bridge = CvBridge()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.4

def image_callback(msg):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        rospy.loginfo("Image received and converted.")
        
        # Resize image to match model input
        original_height, original_width = cv_image.shape[:2]
        resized_image = cv2.resize(cv_image, (640, 640))

        # Convert BGR image to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(rgb_image)
        rospy.loginfo("Object detection performed.")

        # Process detection results
        detections = results.pred[0]  # Access detections directly from results
        rospy.loginfo(f'Detected objects: {detections}')

        # Draw bounding boxes and labels on the image
        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            if conf >= CONFIDENCE_THRESHOLD:
                # Scale bounding box back to original image size
                pt1 = (int(x1 * original_width / 640), int(y1 * original_height / 640))
                pt2 = (int(x2 * original_width / 640), int(y2 * original_height / 640))

                # Print the coordinates of the bounding box
                rospy.loginfo(f'Bounding box coordinates: pt1={pt1}, pt2={pt2}')
                
                color = (0, 255, 0)  # Green color in BGR format
                thickness = 2
                cv2.rectangle(cv_image, pt1, pt2, color, thickness)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.putText(cv_image, label, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        rospy.loginfo("Bounding boxes and labels drawn on image.")

        # Convert OpenCV image back to ROS Image message
        ros_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        rospy.loginfo("Image converted back to ROS Image message.")

        # Publish the image
        detection_image_pub.publish(ros_image)
        rospy.loginfo("Image published.")

    except CvBridgeError as e:
        rospy.logerr(f'CvBridge Error: {e}')
    except AttributeError as e:
        rospy.logerr(f'Attribute Error: {e}')
    except Exception as e:
        rospy.logerr(f'Error: {e}')

# Subscribe to image topic
image_sub = rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
detection_image_pub = rospy.Publisher('/yolov5/detection_image', Image, queue_size=10)

rospy.loginfo("YOLOv5 node is running...")
rospy.spin()
