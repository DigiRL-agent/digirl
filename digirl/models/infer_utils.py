import torch
from PIL import Image
from transformers import AutoProcessor, Blip2VisionModel

class ImageFeatureExtractor:
    def __init__(self):
        # Set device based on CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize and load the BLIP2 model and processor
        self.model = Blip2VisionModel.from_pretrained("/nfs/kun2/users/yifei/.cache/models--Salesforce--blip2-opt-2.7b/snapshots/235c75ea3861136b9dd202c6edc6a7ba285c35e3").to(self.device)
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    def to_feat(self, image: Image.Image):
        """Converts a PIL image to a feature representation using the BLIP2 model.
        
        Args:
            image: A PIL.Image object representing the image to convert.
            
        Returns:
            A tensor representing the image feature.
        """
        with torch.no_grad():
            # Preprocess the image and move to the correct device
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get the image features from the model
            image_features = self.model(**inputs,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        return_dict=False).pooler_output[0]
            #size is 1408
            
            # Detach the tensor from the graph and move it to CPU
            image_features = image_features.detach().cpu()
            
        return image_features
