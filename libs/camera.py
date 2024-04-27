""" 
Reference:
https://github.com/ECI-Robotics/opencv_remote_streaming_processing/
"""

# Import necessary libraries
import cv2
import os
from logging import getLogger

# Create a logger object
logger = getLogger(__name__)

# Define the VideoCamera class
class VideoCamera:
    # Initialize the VideoCamera object
    def __init__(self, input, resize_width, v4l):
        # Set the width to which frames should be resized
        self.resize_width = resize_width
        # If the input is a camera
        if input == "cam":
            # Set the input stream to the default camera
            self.input_stream = 0
            # If v4l is True, use Video for Linux (V4L) for capturing video
            if v4l:
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
            else:
                # Otherwise, use the default video capture
                self.cap = cv2.VideoCapture(self.input_stream)
        else:
            # If the input is not a camera, it should be a video file
            self.input_stream = input
            # Check if the input file exists
            assert os.path.isfile(input), "Specified input file doesn't exist"
            # Capture video from the input file
            self.cap = cv2.VideoCapture(self.input_stream)

        # Read the first frame from the video
        ret, self.frame = self.cap.read()

        # If the frame was read successfully
        if ret:
            # Get the properties of the video capture
            cap_prop = self._get_cap_prop()
            # Log the properties and the resize width
            logger.info(
                "cap_pop:{}, resize_width:{}".format(cap_prop, self.resize_width)
            )
        else:
            # If the frame was not read successfully, log an error message and exit
            logger.error(
                "Please try to start with command line parameters using --v4l if you use RaspCamera"
            )
            os._exit(1)

        # If the height of the frame is greater than the resize width
        if self.frame.shape[0] > self.resize_width:
            # Calculate the scale factor
            scale = self.resize_width / self.frame.shape[1]
            # Resize the frame
            self.frame = cv2.resize(self.frame, dsize=None, fx=scale, fy=scale)

    # When the VideoCamera object is destroyed, release the video capture
    def __del__(self):
        self.cap.release()

    # Get the properties of the video capture
    def _get_cap_prop(self):
        return (
            # Width of the frames
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            # Height of the frames
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            # Frames per second
            self.cap.get(cv2.CAP_PROP_FPS),
        )

    # Get a frame from the video capture
    def get_frame(self, flip_code):
        # Read a frame
        ret, frame = self.cap.read()

        # If the frame is None, return it
        if frame is None:
            return frame

        # If the height of the frame is greater than the resize width
        if frame.shape[0] > self.resize_width:
            # Calculate the scale factor
            scale = self.resize_width / frame.shape[1]
            # Resize the frame
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        # If the frame was read successfully
        if ret:
            # If the input stream is a camera and flip_code is not None
            if self.input_stream == 0 and flip_code is not None:
                # Flip the frame
                frame = cv2.flip(frame, int(flip_code))

            # Return the frame
            return frame
        

"""
The camera.py script is designed to handle video input for processing tasks such as object detection and person re-identification. It defines a VideoCamera class to abstract the complexities of video capture, whether from a live camera or a video file. Here's how it integrates into a broader system and its relevance to the tasks at hand:

1. **Video Input Handling**:
   - The VideoCamera class can initialize with different sources ('cam' for live camera feed or a file path for video files), making it flexible for various deployment scenarios.
   - It uses OpenCV's VideoCapture, which is versatile for reading video from hardware cameras and video files. The ability to switch between these sources is crucial for testing and deploying re-identification systems in different environments.

2. **Video4Linux (V4L) Support**:
   - V4L is specifically used for video capture on Linux systems, providing low-level camera control. This is particularly relevant when using specialized camera hardware such as Raspberry Pi cameras or other Linux-compatible video capture devices.
   - The script checks for the 'v4l' flag to decide whether to use V4L settings, enhancing compatibility and performance on supported systems.

3. **Frame Handling and Resizing**:
   - Upon initialization, it captures the first frame to determine the video properties and ensures the capture device is functioning as expected.
   - Frames are resized based on a specified width (`resize_width`), maintaining efficiency in processing, especially important for real-time object detection and re-identification where processing power may be limited.

4. **Error Handling and Logging**:
   - Proper error handling is implemented to ensure that the system fails gracefully if the video source cannot be opened or read from. This is crucial for deployments in critical systems where continuous operation is required.
   - Logging provides insights into the video properties and operational status, aiding in debugging and system monitoring.

5. **Integration with Object Detection and Re-identification**:
   - For object detection and person re-identification, handling varying frame sizes and formats is essential. This script's capability to standardize frame size and format ensures that the downstream processing can expect consistent input.
   - The method `get_frame` includes an option to flip the frame, which can be useful for correcting camera orientations or for certain processing algorithms that may require a standardized frame orientation.

6. **Potential Improvement with YOLOv9 Integration**:
   - Given that the script is structured to handle generic video input, integrating a more advanced object detection model like YOLOv9 would primarily involve replacing the object detection component in the downstream processing pipeline.
   - YOLOv9, known for its speed and accuracy, could significantly enhance the performance of the re-identification system, especially in challenging environments with fast-moving objects or poor lighting conditions.
   - This integration would involve feeding frames captured by the VideoCamera class directly into the YOLOv9 detection model, then passing detected object coordinates to the re-identification module which could leverage existing features like tracking and feature extraction as implemented in other parts of the system.

In summary, this script is a fundamental component that prepares video data for complex processing tasks such as object detection and person re-identification. It ensures that the video input is compatible, well-structured, and reliably sourced for subsequent analysis steps."
"""
