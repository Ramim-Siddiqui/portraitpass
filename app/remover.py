import io

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


class BackgroundRemover:
    def __init__(self, model_id: str, revision: str, input_size: int, threads: int,
                 device: str = "cpu"):
        self.input_size = input_size
        self.device = torch.device(device)
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, local_files_only=False
        ).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _prediction_tensor(output):
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if isinstance(output, dict):
            output = output.get("preds", output.get("logits", next(iter(output.values()))))
            if isinstance(output, (list, tuple)):
                output = output[-1]
        return output

    def remove_image(self, image: Image.Image) -> Image.Image:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self._prediction_tensor(self.model(tensor)).sigmoid()[0, 0]
        mask = prediction.mul(255).byte().cpu().numpy()
        mask = cv2.resize(mask, image.size, interpolation=cv2.INTER_LANCZOS4)
        # Conservative cleanup: remove isolated noise while retaining hair and soft edges.
        kernel = np.ones((3, 3), np.uint8)
        core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.addWeighted(mask, 0.80, core, 0.20, 0)
        mask = cv2.GaussianBlur(mask, (0, 0), 0.65)
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(mask, mode="L"))
        return rgba

    def remove(self, image: Image.Image) -> bytes:
        rgba = self.remove_image(image)
        output = io.BytesIO()
        rgba.save(output, "PNG", optimize=True)
        return output.getvalue()
