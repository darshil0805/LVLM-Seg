import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
import numpy as np
import cv2
import config

class LVLMSegmenter:
    def __init__(self, model_id=config.MODEL_ID, device=config.DEVICE):
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            attn_implementation="eager" # Eager needed for output_attentions=True usually
        )
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=config.MIN_PIXELS, max_pixels=config.IMAGE_MAX_PIXELS)

    def get_attention_map(self, inputs, original_image_size):
        """
        inputs: dict containing 'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'
        original_image_size: (width, height)
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Forward pass with attentions
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Extract attentions: Tuple of (batch, num_heads, seq_len, seq_len) tensors, one per layer
        # We'll take the average across the middle-to-late layers
        num_layers = len(outputs.attentions)
        # Heuristic: Use layers from 3/4 depth. e.g. for 28 layers, use 14-24. 
        # Let's use the last few layers or a range.
        selected_layers = outputs.attentions[num_layers // 2 : num_layers // 2 + 2] 
        
        # Stack and average across layers: (batch, num_heads, seq_len, seq_len)
        pixel_attentions = torch.stack(selected_layers).mean(dim=0)
        
        # Average across heads: (batch, seq_len, seq_len)
        pixel_attentions = pixel_attentions.mean(dim=1)
        
        # Squeeze batch (batch=1)
        att_map = pixel_attentions[0] # (seq_len, seq_len)
        
        input_ids = inputs['input_ids'][0]
        
        # Find vision token indices
        # Qwen2-VL uses <|vision_start|> and <|vision_end|>
        # Note: The processor handles tokenization. We need to find the token IDs.
        # However, typically 'pixel_values' length determines the vision tokens in the merged sequence? No.
        # In Qwen2-VL, vision tokens are inserted into the sequence.
        # We rely on 'image_grid_thw' to know the shape.
        
        # The visual tokens are usually bounded by special tokens or we can deduce from grid_thw
        # grid_thw is (Time, Height, Width) for the spatial tokens.
        # Sum of T*H*W matches the number of vision tokens.
        
        # Let's find the vision start token id.
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        # Find indices
        vision_starts = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
        vision_ends = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
        
        import pdb; pdb.set_trace()
        
        if len(vision_starts) == 0:
            print("No vision tokens found!")
            return None
            
        # Assuming one image
        start_idx = vision_starts[0].item() + 1 # First token after start
        end_idx = vision_ends[0].item() # Token before end
        
        vision_indices = torch.arange(start_idx, end_idx).to(self.device)
        
        # Text indices: Everything AFTER the image (and excluding padding if any)
        # We want the query text. If valid length is used, we can just take everything after end_idx.
        text_indices = torch.arange(end_idx + 1, len(input_ids)).to(self.device)
        
        # Extract attention: Text (Rows) attending to Vision (Cols)
        # Shape: (num_text_tokens, num_vision_tokens)
        text_to_vision_att = att_map[text_indices][:, vision_indices]
        
        # Average over text tokens to get a single map for the whole query
        # Shape: (num_vision_tokens,)
        # avg_att_map = text_to_vision_att.mean(dim=0)
        # better: max or mean. Let's try mean.
        avg_att_map = text_to_vision_att.mean(dim=0)
        
        # Reshape to grid
        # grid_thw shape is [1, 3] tensor usually [[t, h, w]]
        grid_thw = inputs['image_grid_thw'][0] 
        h, w = grid_thw[1].item(), grid_thw[2].item()
        
        # Qwen2-VL uses 2x2 pooling for vision tokens in the transformer
        # So the number of tokens in the sequence is h//2 * w//2
        # We need to reshape the attention map to (h//2, w//2)
        
        feat_h, feat_w = h // 2, w // 2
        
        if avg_att_map.shape[0] != feat_h * feat_w:
            print(f"Shape mismatch: Vision tokens {avg_att_map.shape[0]}, Grid {h}*{w}={h*w} (Expected {feat_h}*{feat_w}={feat_h*feat_w})")
            return None
            
        att_grid = avg_att_map.view(feat_h, feat_w).float().cpu().numpy()
        
        # Normalize
        att_grid = (att_grid - att_grid.min()) / (att_grid.max() - att_grid.min() + 1e-8)
        
        # Upscale to original image size
        att_upscaled = cv2.resize(att_grid, original_image_size, interpolation=cv2.INTER_LINEAR)
        
        return att_upscaled

