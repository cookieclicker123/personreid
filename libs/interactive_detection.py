# Import necessary libraries
from logging import getLogger
import cv2
import numpy as np
from timeit import default_timer as timer
from libs.tracker import Tracker
import libs.detectors as detectors
import configparser

# Create a logger object
logger = getLogger(__name__)

# Create a configuration parser object
config = configparser.ConfigParser()
# Read the configuration from 'config.ini' file
config.read("config.ini")

# Get the probability threshold for person detection from the configuration
prob_thld_person = eval(config.get("DETECTION", "prob_thld_person"))

# Get the color values for green and skyblue from the configuration
green = eval(config.get("COLORS", "green"))
skyblue = eval(config.get("COLORS", "skyblue"))

# Get the paths for OpenVINO models from the configuration
model_path = config.get("MODELS", "model_path")
model_det = config.get("MODELS", "model_det")
model_reid = config.get("MODELS", "model_reid")

# Define the Detectors class
class Detectors:
    # Initialize the Detectors object
    def __init__(self, devices):
        # Unpack the devices tuple into device_det and device_reid
        self.device_det, self.device_reid = devices
        # Define the models
        self._define_models()
        # Load the detectors
        self._load_detectors()

    # Define the models
    def _define_models(self):
        # Define the path for the person detection model
        fp_path = "FP16-INT8" if self.device_det == "CPU" else "FP16"
        self.model_det = f"{model_path}/{model_det}/{fp_path}/{model_det}.xml"
        # Define the path for the person re-identification model
        fp_path = "FP16-INT8" if self.device_reid == "CPU" else "FP16"
        self.model_reid = f"{model_path}/{model_reid}/{fp_path}/{model_reid}.xml"

    # Load the detectors
    def _load_detectors(self):
        # Load the person detection model
        self.person_detector = detectors.PersonDetection(
            self.device_det, self.model_det
        )
        # Load the person re-identification model
        self.person_id_detector = detectors.PersonReIdentification(
            self.device_reid, self.model_reid
        )


class Detections(Detectors):
    def __init__(self, frame, devices, grid):
        super().__init__(devices)

        # initialize Calculate FPS
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()
        self.prev_frame = frame
        # create tracker instance
        self.tracker = Tracker(self.person_id_detector, frame, grid)

    def _calc_fps(self):
        curr_time = timer()
        exec_time = curr_time - self.prev_time
        self.prev_time = curr_time
        self.accum_time = self.accum_time + exec_time
        self.curr_fps += 1

        if self.accum_time > 1:
            self.accum_time += -1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0

    def draw_bbox(self, frame, box, result, color):
        xmin, ymin, xmax, ymax = box
        size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        xtext = xmin + size[0][0] + 20
        cv2.rectangle(
            frame, (xmin, ymin - 22), (xtext, ymin), green, -1,
        )
        cv2.rectangle(
            frame, (xmin, ymin - 22), (xtext, ymin), green,
        )
        cv2.rectangle(
            frame, (xmin, ymin), (xmax, ymax), green, 1,
        )
        cv2.putText(
            frame,
            result,
            (xmin + 3, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )
        return frame

    def draw_perf_stats(
        self, det_time, det_time_txt, frame, frame_id, is_async, person_counter=None
    ):

        # Draw FPS on top right corner
        self._calc_fps()
        cv2.rectangle(
            frame, (frame.shape[1] - 50, 0), (frame.shape[1], 17), (255, 255, 255), -1
        )
        cv2.putText(
            frame,
            self.fps,
            (frame.shape[1] - 50 + 3, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
        )
        # Draw Real-Time Person Counter on top right corner
        if person_counter is not None:
            cv2.rectangle(
                frame,
                (frame.shape[1] - 50, 17),
                (frame.shape[1], 34),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                frame,
                f"DET: {person_counter}",
                (frame.shape[1] - 50 + 3, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
            )

        # Draw Frame number at bottom corner
        cv2.rectangle(
            frame,
            (frame.shape[1] - 50, frame.shape[0] - 20),
            (frame.shape[1], frame.shape[0]),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            frame,
            frame_id.zfill(5),
            (frame.shape[1] - 40, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
        )

        # Draw performance stats
        if is_async:
            inf_time_message = (
                f"Total Inference time: {det_time * 1000:.3f} ms for async mode"
            )
        else:
            inf_time_message = (
                f"Total Inference time: {det_time * 1000:.3f} ms for sync mode"
            )
        cv2.putText(
            frame,
            inf_time_message,
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 10, 10),
            1,
        )
        if det_time_txt:
            inf_time_message = (
                f"@Detection prob:{prob_thld_person} time: {det_time_txt}"
            )
            cv2.putText(
                frame,
                inf_time_message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 10, 10),
                1,
            )
        return frame

    def get_person_frames(self, persons, frame):

        frame_h, frame_w = frame.shape[:2]
        person_frames = []
        boxes = []

        for person in persons[0][0]:
            box = person[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            xmin, ymin, xmax, ymax = box.astype("int")
            person_frame = frame[ymin:ymax, xmin:xmax]
            person_h, person_w = person_frame.shape[:2]
            # Resizing person_frame will be failed when witdh or height of the person_fame is 0
            # ex. (243, 0, 3)
            if person_h != 0 and person_w != 0:
                boxes.append((xmin, ymin, xmax, ymax))
                person_frames.append(person_frame)

        return person_frames, boxes

    def person_detection(self, frame, is_async, is_det, is_reid, frame_id, show_track):
        # Give the frame_id with tracker instance
        self.tracker.frame_id = frame_id
        self.tracker.show_track = show_track

        # init params
        det_time = 0
        det_time_det = 0
        det_time_reid = 0
        persons = None
        person_frames = None
        boxes = None
        person_counter = 0

        # just return frame when person detection and person reidentification are False
        if not is_det and not is_reid:
            self.prev_frame = self.draw_perf_stats(
                det_time, "Video capture mode", frame, frame_id, is_async
            )
            return self.prev_frame

        if is_async:
            prev_frame = frame.copy()
        else:
            self.prev_frame = frame.copy()

        if is_det or is_reid:

            # ----------- Person Detection ---------- #
            inf_start = timer()
            self.person_detector.infer(self.prev_frame, frame, is_async)
            persons = self.person_detector.get_results(is_async, prob_thld_person)
            inf_end = timer()
            det_time_det = inf_end - inf_start
            det_time_txt = f"person det:{det_time_det * 1000:.3f} ms "

            if persons is None:
                return self.prev_frame

            person_frames, boxes = self.get_person_frames(persons, self.prev_frame)
            person_counter = len(person_frames)

            if is_det and person_frames:
                # ----------- Draw result into the frame ---------- #
                for det_id, person_frame in enumerate(person_frames):
                    confidence = round(persons[0][0][det_id][2] * 100, 1)
                    result = f"{det_id} {confidence}%"
                    # draw bounding box per each person into the frame
                    self.prev_frame = self.draw_bbox(
                        self.prev_frame, boxes[det_id], result, green
                    )

            # ----------- Person ReIdentification ---------- #
            if is_reid:
                inf_start = timer()
                self.prev_frame = self.tracker.person_reidentification(
                    self.prev_frame, person_frames, boxes
                )
                inf_end = timer()
                det_time_reid = inf_end - inf_start
                det_time_txt = det_time_txt + f"reid:{det_time_reid * 1000:.3f} ms"

            if person_frames is None:
                det_time_txt = "No persons detected"

        det_time = det_time_det + det_time_reid
        frame = self.draw_perf_stats(
            det_time,
            det_time_txt,
            self.prev_frame,
            frame_id,
            is_async,
            person_counter=str(person_counter),
        )

        if is_async:
            self.prev_frame = prev_frame

        return frame


"""
This script implements a class-based approach to person re-identification, focusing particularly on object detection using OpenVINO toolkit models and feature extraction. Here's a breakdown of how the object detection process is executed:

1. **Configuration and Initialization**: The script starts by importing necessary libraries and setting up a configuration parser to read settings from a 'config.ini' file. These settings include model paths, device information, and visual attributes like colors.

2. **Detectors Class**:
   - **Initialization**: Initializes with a specific device setup for detection and re-identification.
   - **Model Definitions**: Specifies paths to the XML model files for detection and re-identification, which are likely configured for use with Intel's OpenVINO toolkit.
   - **Load Detectors**: Instantiates the object detection and re-identification models. The object detection is specifically noted as 'PersonDetection', which implies the model is fine-tuned or designated for detecting individuals.

3. **Detection Implementation in Detections Class**:
   - Inherits from the `Detectors` class, indicating it uses the defined models for its operations.
   - **Frame Processing**: Each frame received by the system is processed to calculate the frames per second (FPS), draw bounding boxes around detected persons, and calculate and display performance statistics.
   - **Object Detection and Tracking**: Utilizes the 'PersonDetection' model to detect persons in each frame and tracks these detections over time using a 'Tracker' instance. This tracking helps maintain identity consistency across frames, which is crucial for re-identification.

4. **Performance Considerations**:
   - The system calculates FPS to monitor and display processing efficiency.
   - It draws bounding boxes and other performance metrics directly on the video frames for real-time monitoring.

5. **Potential Issues with Current Setup**:
   - **Detection Speed and Accuracy**: The current detection model may not be the fastest or most accurate, particularly in environments with frequent scene changes, as typical in TV shows or locations with multiple non-overlapping camera views.
   - **Ghost Boxes Problem**: When the ID of a person is lost (due to occlusion or scene switches), the system might display lingering 'ghost' boxes waiting for the person to reappear, which can lead to inaccuracies or false identifications in dynamic environments.

**Proposed Solution - Integration with YOLOv9**:
   - To address these issues, integrating YOLOv9 as the object detection model could be considered. YOLOv9, being a part of the more recent iterations of the YOLO (You Only Look Once) family, is designed for faster real-time object detection with substantial improvements in accuracy.
   - **Steps to Integrate YOLOv9**:
     1. Replace the current detection model with a pre-trained YOLOv9 model adjusted for person detection. This involves modifying the `_load_detectors` method to load YOLOv9 instead of the OpenVINO model.
     2. Adapt the input and output processing in the script to match the requirements of YOLOv9, such as adjusting the input frame dimensions and parsing the output tensors for bounding box coordinates and class predictions.
     3. Ensure the YOLOv9 model is optimized for the devices being used, potentially leveraging NVIDIA GPUs for better performance if available.
   - By combining YOLOv9's robust detection capabilities with the existing feature extraction mechanisms of this code, the system could more effectively handle scene changes and maintain accurate re-identifications across varied and challenging video sequences.

This enhanced approach could potentially improve the system's ability to handle real-world scenarios where subjects frequently enter and exit the scene, thus minimizing issues with ghost detections and improving overall accuracy and reliability of the person re-identification system.
"""