import numpy as np
import cv2
import traceback
from scipy.spatial import distance

def resize_frame(frame, height):
    """Resize the frame to a specified height while maintaining the aspect ratio."""
    try:
        scale = height / frame.shape[1]  # Calculate the scaling factor based on the desired height and the current width.
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)  # Apply the scaling factor to resize the frame.
    except ZeroDivisionError as e:
        traceback.print_exc()  # Print stack trace if there is a division by zero error.
    except cv2.error as e:
        traceback.print_exc()  # Print stack trace if OpenCV encounters an error during resizing.
    return frame  # Return the resized frame.

#updated to handle 1D and 2D arrays
def cos_similarity(X, Y):
    """Calculate the cosine similarity between two arrays X and Y."""
    if X.ndim == 1 and Y.ndim == 1:
        dot_product = np.dot(X, Y)
        magnitude = np.linalg.norm(X) * np.linalg.norm(Y)
        if magnitude == 0:
            return 0
        return dot_product / magnitude
    else:
        X = X if X.ndim > 1 else X[None, :]
        Y = Y.T if Y.ndim > 1 else Y[None, :].T
        dot_product = np.dot(X, Y)
        magnitude = np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(Y, axis=0)
        magnitude_zero = magnitude == 0
        if np.any(magnitude_zero):
            dot_product[magnitude_zero] = 0
            magnitude[magnitude_zero] = 1  # To avoid division by zero
        return dot_product / magnitude


def get_euclidean_distance(x, Y):
    """Calculate the Euclidean distance between a vector x and each vector in matrix Y."""
    return np.linalg.norm(x - Y, axis=1)  # Compute distance for each pair of vectors.

def get_iou2(box, boxes):
    """Calculate the Intersection over Union (IoU) between a given box and multiple other boxes."""
    ximin = np.maximum(box[0], boxes[:, 0])  # Calculate the x-coordinate of the left side of the intersection.
    yimin = np.maximum(box[1], boxes[:, 1])  # Calculate the y-coordinate of the top side of the intersection.
    ximax = np.minimum(box[2], boxes[:, 2])  # Calculate the x-coordinate of the right side of the intersection.
    yimax = np.minimum(box[3], boxes[:, 3])  # Calculate the y-coordinate of the bottom side of the intersection.
    inter_width = ximax - ximin  # Width of the intersection area.
    inter_height = yimax - yimin  # Height of the intersection area.
    inter_area = np.maximum(inter_width, 0) * np.maximum(inter_height, 0)  # Area of the intersection.
    box_area = (box[2] - box[0]) * (box[3] - box[1])  # Area of the first box.
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # Area of each box in 'boxes'.
    union_area = box_area + boxes_area - inter_area  # Total area covered by both the original and other boxes.
    return inter_area / union_area  # Return the IoU value for each comparison.



def get_iou(box1: tuple, box2: tuple) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    # Calculate the coordinates of the intersection rectangle
    ximin = max(box1[0], box2[0])
    yimin = max(box1[1], box2[1])
    ximax = min(box1[2], box2[2])
    yimax = min(box1[3], box2[3])
    inter_width = ximax - ximin
    inter_height = yimax - yimin
    inter_area = max(inter_width, 0) * max(inter_height, 0)
    # Calculate the area of each box and the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area  # IoU calculation

def get_mahalanobis_distance(center, track_points):
    """Calculate the Mahalanobis distance between a center point and a set of track points."""
    cov = np.cov(track_points.T)  # Calculate the covariance matrix of track points
    # Return the Mahalanobis distance using the last track point
    return distance.mahalanobis(track_points[-1], center, np.linalg.pinv(cov))

def affine_translation(box: tuple, top_left: tuple = (0, 0)) -> tuple:
    """Translate a bounding box by subtracting the coordinates of a top-left point."""
    translation_matrix = np.eye(3)  # Identity matrix for affine transformation
    translation_matrix[0][2] = -1 * (box[0] - top_left[0])
    translation_matrix[1][2] = -1 * (box[1] - top_left[1])
    # Apply transformation to box coordinates
    min = np.array([box[0], box[1], 1]).reshape(3, 1)
    max = np.array([box[2], box[3], 1]).reshape(3, 1)
    min = translation_matrix @ min
    max = translation_matrix @ max
    return (min[0][0], min[1][0], max[0][0], max[1][0])

def get_standard_deviation(x: list) -> tuple:
    """Calculate the mean and standard deviation of a list of values."""
    mean = np.mean(x)
    std = np.std(x)
    return mean, std

def get_box_coordinates(prev_box, center) -> tuple:
    """Calculate new bounding box coordinates based on a previous box and a new center point."""
    box_w, box_h = prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]
    xmin = center[0] - (box_w / 2)
    ymin = center[1] - (box_h / 2)
    xmax = xmin + box_w
    ymax = ymin + box_h
    return (xmin, ymin, xmax, ymax)



"""
The 'utils.py' script provides a suite of utility functions essential for image processing, tracking computations, and geometric transformations within a person re-identification system. These utilities are integral to handling frame resizing, calculating similarity measures, and determining spatial relationships between detected objects. Here is an overview of the key functionalities:

1. **Frame Resizing (resize_frame)**:
   - Adjusts the size of an input frame to a specified height while maintaining aspect ratio. This function is crucial for standardizing the input size for consistent processing, especially when feeding frames into neural network models that require input dimensions to be uniform.

2. **Cosine Similarity (cos_similarity)**:
   - Calculates the cosine similarity between two sets of feature vectors. This measure helps in determining the similarity between the features extracted from different detections, which is fundamental for matching detections across frames in the tracking process.

3. **Euclidean Distance (get_euclidean_distance)**:
   - Computes the Euclidean distance between vectors, typically used to measure the physical distance between detected points in a frame. This function supports operations such as determining how far a tracked object has moved between consecutive frames, aiding in motion analysis and trajectory prediction.

4. **Intersection over Union (get_iou, get_iou2)**:
   - These functions calculate the Intersection over Union (IoU) of bounding boxes, a critical measure in object detection and tracking to evaluate how much one bounding box overlaps with another. It is used to manage occlusions and to decide whether detections correspond to the same object.

5. **Mahalanobis Distance (get_mahalanobis_distance)**:
   - Measures the distance between a point and a distribution, used for more sophisticated track management where the distance is calculated considering the variability of the track's trajectory. This can be particularly effective in distinguishing between closely moving objects and improving track stability.

6. **Affine Translation (affine_translation)**:
   - Translates bounding boxes based on a given reference point, useful for alignment and relative positioning tasks in image processing pipelines, ensuring that spatial transformations maintain their relational integrity across different operations.

7. **Standard Deviation Calculation (get_standard_deviation)**:
   - Computes the mean and standard deviation of a data set, often used in statistical analysis within tracking algorithms to understand the variability and confidence of the tracking parameters.

8. **Bounding Box Coordinate Calculation (get_box_coordinates)**:
   - Generates new bounding box coordinates based on a central point and the dimensions of a previous box. This is especially useful in predictive tracking models like Kalman filters where the next position of an object needs to be estimated.

**Potential Enhancements with Advanced Detection Models and Feature Extraction Techniques**:
   - By integrating high-performance detection models like YOLO, the initial detection accuracy can be significantly improved, leading to better input quality for the tracking and re-identification systems. These models are faster and more accurate, especially in diverse environments.
   - Enhancing feature extraction methods by adopting more sophisticated neural network architectures designed specifically for re-identification could yield richer and more discriminative features. This could include using deep metric learning techniques that focus on minimizing intra-class variations and maximizing inter-class differences, which would be vital in improving match accuracies across non-overlapping camera feeds.
   - Improvements in geometric transformation functions and the incorporation of machine learning-based predictive models could further refine the positional estimations and tracking consistency, especially in complex scenarios involving frequent scene changes or varying angles of view.

In summary, the 'utils.py' script is foundational to the functioning of a re-identification system, providing the necessary tools for processing and analyzing video data effectively. By upgrading these utility functions with cutting-edge algorithms and integrating them with advanced object detection and feature extraction models, the robustness and accuracy of the re-identification system can be significantly enhanced, making it adaptable to a wider range of video surveillance and tracking applications."
"""
