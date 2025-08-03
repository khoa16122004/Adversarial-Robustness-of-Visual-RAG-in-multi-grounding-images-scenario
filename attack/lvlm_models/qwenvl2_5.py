import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import List, Union
from PIL import Image
import math
import torchvision.transforms as T

class QwenVL:
    def __init__(self, pretrained="Qwen2.5-VL-7B-Instruct", device="cuda"):
        model_path = f"Qwen/{pretrained}"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.model.eval()

    def __call__(self, qs, img_files):

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": qs}]
                          + [{"type": "image", "image": img} for img in img_files]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = self._process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
            )

        # Remove prompt tokens from outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
    def compute_log_prob(self, question: str, imgs: List[Image.Image], answer: str) -> float:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": question}]
                    + [{"type": "image", "image": img} for img in imgs]
        }]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        full_input = prompt + answer
        tokenized = self.processor(
            text=[full_input],
            images=imgs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        input_ids = tokenized.input_ids
        labels = input_ids.clone()

        with torch.no_grad():
            prompt_only = self.processor(
                text=[prompt],
                images=imgs,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            prompt_len = prompt_only.input_ids.shape[1]

        labels[0, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(
                **tokenized,
                labels=labels
            )
            loss = outputs.loss

        num_answer_tokens = input_ids.shape[1] - prompt_len
        total_log_prob = -loss.item() * num_answer_tokens
        prob = math.exp(total_log_prob)

        return prob

    def _process_vision_info(self, messages):
        images = []
        videos = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    images.append(content["image"])
                elif content["type"] == "video":
                    videos.append(content["video"])
        return images, videos
    
    

