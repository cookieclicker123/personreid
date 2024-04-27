# Import necessary libraries
import cv2
import numpy as np
from logging import getLogger

# Import OpenVINO runtime Core, get_version and alias it as ov
from openvino.runtime import Core
import openvino.runtime as ov
from openvino.runtime import get_version

# Create a logger for this module
logger = getLogger(__name__)

# Define a base class for detection tasks
class BaseDetection(object):
    def __init__(self, device, model_xml, detection_of):
        # Initialize the inference engine
        ie = Core()
        # Read the network and corresponding weights from file
        model = ie.read_model(model=model_xml)

        # Compile the model for the specified device (CPU, GPU, MYRIAD, etc.)
        self.compiled_model = ie.compile_model(model=model, device_name=device)

        # Get the input layer of the model
        self.input_layer_ir = model.input(0)
        # Get the shape of the input layer
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)

        # Log the model loading info
        logger.info(
            f"Loading {device} model to the {detection_of} ... version:{get_version()}"
        )

    def preprocess(self, frame):
        """
        Preprocess the input frame:
        1. Resize the frame to match the input shape of the model
        2. Convert the color space from BGR to RGB
        3. Transpose the frame dimensions from HWC to CHW
        4. Add an extra dimension to the frame
        """
        resized_frame = cv2.resize(frame, self.shape)
        resized_frame = cv2.cvtColor(
            np.array(resized_frame), cv2.COLOR_BGR2RGB)
        resized_frame = resized_frame.transpose((2, 0, 1))
        resized_frame = np.expand_dims(
            resized_frame, axis=0).astype(np.float32)
        return resized_frame


# Define a class for person detection
class PersonDetection(BaseDetection):
    def __init__(self, device, model_xml):
        # Initialize the base class with the model details
        super().__init__(device, model_xml, "Person Detection")

        # Create 2 inference requests for async inference
        self.curr_request = self.compiled_model.create_infer_request()
        self.next_request = self.compiled_model.create_infer_request()

    def infer(self, frame, next_frame, is_async):
        """
        Perform inference on the input frame.
        If is_async is True, perform inference on the next frame,
        otherwise wait for the current inference to complete and then perform inference on the current frame.
        """
        if is_async:
            resized_frame = self.preprocess(next_frame)
            self.next_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the "next" inference request
            self.next_request.start_async()

        else:
            self.curr_request.wait_for(-1)
            resized_frame = self.preprocess(frame)
            self.curr_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the current inference request
            self.curr_request.start_async()

    def get_results(self, is_async, prob_threshold_person):
        """
        Get the inference results.
        If the current inference is complete, get the output tensor and filter the results based on the probability threshold.
        If is_async is True, swap the current and next inference requests.
        """
        persons = None
        if self.curr_request.wait_for(-1) == 1:
            res = self.curr_request.get_output_tensor(0).data
            persons = res[0][:, np.where(
                res[0][0][:, 2] > prob_threshold_person)]

        if is_async:
            self.curr_request, self.next_request = self.next_request, self.curr_request

        return persons


# Define a class for person re-identification
class PersonReIdentification(BaseDetection):
    def __init__(self, device, model_xml):
        # Initialize the base class with the model details
        super().__init__(device, model_xml, "Person re-identifications")
        # Create an inference request
        self.infer_request = self.compiled_model.create_infer_request()

    def infer(self, person_frame):
        # Preprocess the input frame and perform inference
        resized_frame = self.preprocess(person_frame)
        self.infer_request.set_tensor(
            self.input_layer_ir, ov.Tensor(resized_frame))
        self.infer_request.infer()

    def get_results(self):
        """
        Get the inference results.
        The output is a blob with the shape [1, 256, 1, 1] named descriptor,
        which can be compared with other descriptors using the cosine distance.
        """
        res = self.infer_request.get_output_tensor(0).data
        feature_vec = res.reshape(1, 256)
        return feature_vec
    


"""
The 'detectors.py' script is designed to facilitate object detection and person re-identification tasks using Intel's OpenVINO toolkit. This script includes classes that encapsulate the processes needed to handle deep learning model inference, specifically tailored for person detection and re-identification. Below is a detailed explanation of the main components and their functionality:

1. **BaseDetection Class**:
   - **Initialization**: This class serves as a foundation for all specific detection tasks. It initializes the OpenVINO inference engine, loads the specified model, and sets up the necessary parameters for inference.
     - An instance of the OpenVINO Core is created to manage all activities related to model management and inference.
     - The model, specified by its XML file path, is loaded into the engine. The model and its weights are read from the file and then compiled for a specified device (CPU, GPU, MYRIAD, etc.), optimizing the model for that device.
     - The input layer of the model is accessed to determine the required shape for input frames, essential for preprocessing steps before inference.
   - **Preprocessing**: Converts input frames to the format required by the neural network. This involves resizing the frame to match the model's input dimensions, converting the color space from BGR to RGB (as OpenVINO models typically expect RGB input), and rearranging the frame data from Height x Width x Channels (HWC) to Channels x Height x Width (CHW). An additional batch dimension is added to the data since OpenVINO expects inputs in this format.

2. **PersonDetection Class**:
   - **Inherits from BaseDetection**: It uses the foundational setup from the BaseDetection class and includes additional methods specific to person detection.
   - **Async Inference Management**: Manages two inference requests to allow asynchronous processing. This means while one frame is being processed, the next frame can be prepared, significantly speeding up the inference process.
   - **Inference Execution**: Depending on the mode (asynchronous or not), it processes either the current frame or prepares the next frame for inference. The frame data is passed to the model after preprocessing.
   - **Result Handling**: After inference, the results are filtered based on a confidence threshold to determine which detections are likely to represent actual persons. This filtering is crucial for reducing false positives and ensuring that subsequent processing stages, such as re-identification, operate on relevant data.

3. **PersonReIdentification Class**:
   - **Inherits from BaseDetection**: Similar to the PersonDetection class but tailored for re-identifying persons based on the features extracted by the person re-identification model.
   - **Feature Extraction**: After preprocessing the input frame (or cropped image of a person), the model processes it to extract a feature vector. This vector represents the person in a compact form, suitable for comparing with other vectors using distance metrics (like cosine similarity) to determine if two images represent the same person.

**Integration in a Re-Identification System**:
- These classes provide the necessary functionality to capture video, detect persons, and re-identify them across different frames or even different cameras. This capability is particularly useful in surveillance, retail analytics, and any application where tracking individuals across scenes is required.
- The modular design allows for easy integration and customization of the detection and re-identification processes, enabling developers to adapt the system to specific needs, such as adjusting confidence thresholds or changing the preprocessing pipeline to better suit different environmental conditions or camera setups.

**Potential Enhancements**:
- Integration with more advanced object detection models or frameworks, like YOLO or SSD, could be explored to improve detection accuracy and speed.
- Enhancements in the preprocessing steps, such as using more sophisticated image normalization techniques, could further improve model performance, especially under varying lighting conditions.
"""
