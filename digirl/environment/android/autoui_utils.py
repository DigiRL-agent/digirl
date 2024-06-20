from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Union
from transformers import Blip2VisionModel, AutoProcessor, Blip2Model
import torch
from PIL import Image

class ImageFeatureExtractor:
    def __init__(self, device):
        # Set device based on CUDA availability
        self.device = device
        
        # Initialize and load the BLIP2 model and processor
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b").cpu()
        self.model.language_model = None
        # self.model = self.model.to(self.device)
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
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            
            # Get the image features from the model
            image_features = self.model.get_image_features(**inputs).pooler_output[0]
            
            # Detach the tensor from the graph and move it to CPU
            image_features = image_features.detach().cpu()
            
        return image_features

# class ImageFeatureExtractor:
#     def __init__(self, device):
#         # Set device based on CUDA availability
#         self.device = device
        
#         # Initialize and load the BLIP2 model and processor
#         self.model = Blip2VisionModel.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
#         self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

#     def to_feat(self, image: Image.Image):
#         """Converts a PIL image to a feature representation using the BLIP2 model.
        
#         Args:
#             image: A PIL.Image object representing the image to convert.
            
#         Returns:
#             A tensor representing the image feature.
#         """
#         with torch.no_grad():
#             # Preprocess the image and move to the correct device
#             inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
#             # Get the image features from the model
#             image_features = self.model(**inputs,
#                                         output_attentions=False,
#                                         output_hidden_states=False,
#                                         return_dict=True).pooler_output[0]
#             #size is 1408
            
#             # Detach the tensor from the graph and move it to CPU
#             image_features = image_features.detach().cpu()
            
#         return image_features

class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7

@dataclass
class AndroidAction():
    action_type: ActionType
    touch_point: Tuple[float, float] = None
    lift_point: Tuple[float, float] = None
    typed_text: str = None

    def __str__(self):
        # Construct the basic action type string.
        components = [f"Action Type: {self.action_type.name}"]

        # Format and add touch_point if it's not None.
        if self.touch_point:
            touch_point_str = f"({self.touch_point[0]:.4f}, {self.touch_point[1]:.4f})"
            components.append(f"Touch Point: {touch_point_str}")

        # Format and add lift_point if it's not None.
        if self.lift_point:
            lift_point_str = f"({self.lift_point[0]:.4f}, {self.lift_point[1]:.4f})"
            components.append(f"Lift Point: {lift_point_str}")

        # Add typed_text if it's not None.
        if self.typed_text:
            components.append(f"Typed Text: '{self.typed_text}'")

        # Join all components into a single string.
        return ", ".join(components)

    def to_act(self):
        pass


def cogagent_translate_action(out):
    raw_action = out
    try:
        raw_action = raw_action.split('Grounded Operation:')[1]
        action = raw_action.split(" ")[0]
        if action == 'tap':
            numbers = raw_action.split('[[')[1].split(',')
            x = int(numbers[0])
            y = int(numbers[1].split(']]')[0])
            touch_point = (x/1000, y/1000)
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=touch_point)
        elif "type" in action:
            text = raw_action.split('"')[1]
            return AndroidAction(action_type=ActionType.Type, typed_text=text)
        elif "press home" in raw_action:
            return AndroidAction(action_type=ActionType.GoHome)
        elif "press back" in raw_action:
            return AndroidAction(action_type=ActionType.GoBack)
        elif "press enter" in raw_action:
            return AndroidAction(action_type=ActionType.Enter)
        elif "task complete" in raw_action:
            return AndroidAction(action_type=ActionType.TaskComplete)
        elif "task impossible" in raw_action:
            return AndroidAction(action_type=ActionType.TaskImpossible)
        elif "swipe up" in raw_action:
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.5), lift_point=(0.5, 0.2))
        elif "swipe down" in raw_action:
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.2), lift_point=(0.5, 0.5))
        elif "swipe left" in raw_action:
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.8, 0.5), lift_point=(0.2, 0.5))
        elif "swipe right" in raw_action:
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.2, 0.5), lift_point=(0.8, 0.5))
        else:
            print(f"Action {raw_action} not supported yet.")
            return AndroidAction(action_type=ActionType.Idle)
    except Exception as e:
        print(f"Action {raw_action} Parsing Error: {e}")
        return AndroidAction(action_type=ActionType.Idle)

def autoui_translate_action(out):
    action_str = out.split("Action Decision: ")[1]
    action_type, touch_point_1, touch_point_2, lift_point_1, lift_point_2, typed_text = action_str.split(", ")
    touch_point = touch_point_1 + ", " + touch_point_2
    lift_point = lift_point_1 + ", " + lift_point_2
    try:
        action_type = action_type.split(": ")[1].strip('"')
        if action_type == 'DUAL_POINT':
            touch_point_yx = touch_point.split(": ")[1].strip('[]"')
            touch_point_yx = [float(num) for num in touch_point_yx.split(", ")]
            lift_point_yx = lift_point.split(": ")[1].strip('[]"')
            lift_point_yx = [float(num) for num in lift_point_yx.split(", ")]
            return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point_yx[::-1], lift_point=lift_point_yx[::-1])
        elif action_type == 'TYPE':
            text = typed_text.split(": ")[1].strip('"')
            return AndroidAction(action_type=ActionType.Type, typed_text=text)
        elif action_type == 'PRESS_HOME':
            return AndroidAction(action_type=ActionType.GoHome)
        elif action_type == 'PRESS_BACK':
            return AndroidAction(action_type=ActionType.GoBack)
        elif action_type == 'PRESS_ENTER':
            return AndroidAction(action_type=ActionType.Enter)
        elif action_type == 'STATUS_TASK_COMPLETE':
            return AndroidAction(action_type=ActionType.TaskComplete)
        elif action_type == 'TASK_IMPOSSIBLE':
            return AndroidAction(action_type=ActionType.TaskImpossible)
        else:
            print(f"Action {out} not supported yet.")
            return AndroidAction(action_type=ActionType.Idle)
    except Exception as e:
        print(f"Action {out} Parsing Error: {e}")
        return AndroidAction(action_type=ActionType.Idle)

def to_autoui(act: AndroidAction):
    if act.action_type == ActionType.DualPoint:
        return f'"action_type": "DUAL_POINT", "touch_point": "[{act.touch_point[1]:.4f}, {act.touch_point[0]:.4f}]", "lift_point": "[{act.lift_point[1]:.4f}, {act.lift_point[0]:.4f}]", "typed_text": ""'
    elif act.action_type == ActionType.Type:
        return f'"action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "{act.typed_text}"'
    elif act.action_type == ActionType.GoBack:
        return f'"action_type": "PRESS_BACK", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.GoHome:
        return f'"action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.Enter:
        return f'"action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.TaskComplete or act.action_type == ActionType.TaskImpossible:
        return f'"action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    else:
        print(f"Action {act} not supported yet.")
        return ""

def autoui_prepare_prompt(task, history):
        prompt = "Previous Actions: "
        for act in history[-1:]:
            prompt += f"{to_autoui(act)} "
        prompt += f"Goal: {task}</s>"
        return prompt