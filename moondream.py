from importlib.util import find_spec
from typing import List, Dict, Any, Optional, Union
import fiftyone as fo
from fiftyone import Model
from fiftyone.core.labels import Detection, Detections, Keypoint, Keypoints
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

MOONDREAM_OPERATIONS = {
    "caption": {
        "params": {"length": ["short", "normal"]},
        "required": ["length"],
    },
    "query": {
        "params": {"query_text": str},
        "required": ["query_text"],
    },
    "detect": {
        "params": {"object_type": str},
        "required": ["object_type"],
    },
    "point": {
        "params": {"object_type": str},
        "required": ["object_type"],
    }
}

def moondream_activator():
    """Check if required dependencies are installed."""
    return find_spec("transformers") is not None and find_spec("torch") is not None

def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Moondream2(Model):
    """A FiftyOne model for running the Moondream2 model on images.
    
    Args:
        operation (str): Type of operation to perform
        model_name (str): Name of the model to load from HuggingFace
        revision (str, optional): Model revision/tag to use
        device (str, optional): Device to run the model on ('cuda', 'mps', or 'cpu')
        **kwargs: Operation-specific parameters
    """

    def __init__(
        self, 
        operation: str,
        model_name: str = "vikhyatk/moondream2",
        revision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        self.operation = operation
        
        # Validate operation
        if operation not in MOONDREAM_OPERATIONS:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {list(MOONDREAM_OPERATIONS.keys())}")
        
        # Validate required parameters
        required_params = MOONDREAM_OPERATIONS[operation]["required"]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter '{param}' for operation '{operation}'")
        
        self.params = kwargs

        # Set device
        self.device = device or get_device()

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device_map={"": self.device},
            torch_dtype=torch.float16
        )

    @property
    def media_type(self):
        return "image"

    def _convert_to_detections(self, boxes: List[Dict[str, float]], label: str) -> Detections:
        """Convert Moondream2 detection output to FiftyOne Detections.
        
        Args:
            boxes: List of bounding box dictionaries
            label: Object type label
            
        Returns:
            FiftyOne Detections object
        """
        detections = []

        for box in boxes:
            detection = Detection(
                label=label,
                bounding_box=[
                    box["x_min"],
                    box["y_min"],
                    box["x_max"] - box["x_min"],  # width
                    box["y_max"] - box["y_min"]   # height
                ]
            )

            detections.append(detection)
        
        return Detections(detections=detections)

    def _convert_to_keypoints(self, points: List[Dict[str, float]], label: str) -> Keypoints:
        """Convert Moondream2 point output to FiftyOne Keypoints.
        
        Args:
            points: List of point dictionaries
            label: Object type label
            
        Returns:
            FiftyOne Keypoints object
        """
        keypoints = []

        for idx, point in enumerate(points):

            keypoint = Keypoint(
                label=f"{label}_{idx+1}",
                points=[[point["x"], point["y"]]]
            )

            keypoints.append(keypoint)
        
        return Keypoints(keypoints=keypoints)

    def _predict_caption(self, image: Image.Image) -> Dict[str, str]:
        """Generate a caption for an image.
        
        Args:
            image: PIL image
            
        Returns:
            dict: Caption result
        """
        result = self.model.caption(image, length=self.params["length"])["caption"]

        return {"caption": result}

    def _predict_query(self, image: Image.Image) -> Dict[str, str]:
        """Answer a visual query about an image.
        
        Args:
            image: PIL image
            
        Returns:
            dict: Query answer
        """
        result = self.model.query(image, self.params["query_text"])["answer"]

        return {"answer": result}

    def _predict_detect(self, image: Image.Image) -> Dict[str, Detections]:
        """Detect objects in an image.
        
        Args:
            image: PIL image
            
        Returns:
            dict: Detection results
        """
        result = self.model.detect(image, self.params["object_type"])["objects"]

        detections = self._convert_to_detections(result, self.params["object_type"])

        return {"detections": detections}

    def _predict_point(self, image: Image.Image) -> Dict[str, Keypoints]:
        """Identify point locations of objects in an image.
        
        Args:
            image: PIL image
            
        Returns:
            dict: Keypoint results
        """
        result = self.model.point(image,self.params["object_type"])["points"]

        keypoints = self._convert_to_keypoints(result, self.params["object_type"])

        return {"keypoints": keypoints}

    def _predict(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single image with Moondream2.
        
        Args:
            image: PIL image
            
        Returns:
            dict: Operation results
        """
        prediction_methods = {
            "caption": self._predict_caption,
            "query": self._predict_query,
            "detect": self._predict_detect,
            "point": self._predict_point
        }
        
        predict_method = prediction_methods.get(self.operation)

        if predict_method is None:
            raise ValueError(f"Unknown operation: {self.operation}")
            
        return predict_method(image)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Process an image array with Moondream2.
        Args:
        
            image: numpy array image
            
        Returns:
            dict: Operation results
        """
        pil_image = Image.fromarray(image)
        return self._predict(pil_image)

def run_moondream_model(
    dataset: fo.Dataset,
    operation: str,
    output_field: str,
    model_name: str = "vikhyatk/moondream2",
    revision: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> None:
    """Apply Moondream2 operations to a FiftyOne dataset.
    
    Args:
        dataset: FiftyOne dataset to process
        operation: Type of operation to perform
        output_field: Field to store results in
        model_name: Name of the model to load from HuggingFace
        revision: Model revision/tag to use
        device: Device to run the model on
        **kwargs: Operation-specific parameters
    """
    model = Moondream2(
        operation=operation,
        model_name=model_name,
        revision=revision,
        device=device,
        **kwargs
    )
    dataset.apply_model(model, label_field=output_field)