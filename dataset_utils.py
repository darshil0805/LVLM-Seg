
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import config

class RefCOCODataset(Dataset):
    def __init__(self, split="val", model_id=config.MODEL_ID):
        self.dataset = load_dataset("lmms-lab/RefCOCO", split=split)
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=config.MIN_PIXELS, max_pixels=config.IMAGE_MAX_PIXELS)
        
    def __len__(self):
        # Limit to 50 samples for quick testing/demo
        return min(len(self.dataset), 50) 
        # return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # The dataset structure usually has 'image' (PIL) and 'sentences' (list of dicts or strings)
        # We need to check the specific structure of "lmms-lab/RefCOCO".
        # Assuming standard RefCOCO format: 'image', 'sentences' -> [{'raw': 'text', ...}]
        
        image = item['image']
        # Convert to RGB if not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        

        # Get the first referring expression
        # Adjust based on actual dataset features. 
        # If 'sentences' key exists:
        # if 'sentences' in item:
        #     # Check if it's a list of strings or dicts
        #     first_sent = item['sentences'][0]
        #     if isinstance(first_sent, dict) and 'raw' in first_sent:
        #         ref_exp = first_sent['raw']
        #     elif isinstance(first_sent, dict) and 'sent' in first_sent:
        #         ref_exp = first_sent['sent']
        #     else:
        #         ref_exp = str(first_sent)
        # elif 'sentences_raw' in item:
        #      ref_exp = item['sentences_raw'][0]
        # else:
        #      # Fallback or check other keys. Some HF datasets use 'caption'
        #     ref_exp = "object"

        if 'answer' not in item or len(item['answer']) == 0:
            ref_exp = "object"
        else:
            ref_exp = item['answer'][0]

        # Prepare input for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": f"Find {ref_exp}."},
                ],
            }
        ]
        
        # Preprocessing
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Keep batch dimension as we are not using a DataLoader that collates
        # inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return {
            "inputs": inputs,
            "image": image, # Return original image for visualization
            "text": ref_exp
        }

def get_dataloader(batch_size=1):
    # Only batch_size=1 supported for now due to variable image sizes/tensor shapes in naive collation
    dataset = RefCOCODataset()
    return dataset

