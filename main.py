
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from qwen_vl_utils import process_vision_info
import config
from dataset_utils import get_dataloader
from segmentation import LVLMSegmenter

def apply_heatmap(image_pil, attention_map):
    # image_pil: RGB PIL image
    # attention_map: numpy array (H, W), normalized [0, 1]
    
    img = np.array(image_pil)
    
    # Convert attention map to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    alpha = 0.5
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlay)

def main():
    print(f"Loading model: {config.MODEL_ID}")
    segmenter = LVLMSegmenter()
    
    print("Loading dataset...")
    dataset = get_dataloader()
    
    output_dir = config.OUTPUT_DIR
    print(f"Saving outputs to {output_dir}")
    
    # Process a few samples
    for i in range(len(dataset)):
        print(f"Processing sample {i+1}/{len(dataset)}...")
        sample = dataset[i]
        inputs = sample['inputs']
        original_image = sample['image']
        text_query = sample['text']
        
        # Get attention map
        att_map = segmenter.get_attention_map(inputs, original_image.size)
        
        if att_map is not None:
            # Save raw attention map
            # plt.imsave(os.path.join(output_dir, f"sample_{i}_att.png"), att_map, cmap='jet')
            
            # Create overlay
            overlay_img = apply_heatmap(original_image, att_map)
            overlay_img.save(os.path.join(output_dir, f"sample_{i}_overlay.png"))
            
            # Save original with text
            plt.figure(figsize=(10, 10))
            plt.imshow(original_image)
            plt.title(f"Query: {text_query}")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"sample_{i}_ref.png"))
            plt.close()
            
            print(f"Saved sample {i}")
        else:
            print(f"Failed to generate attention map for sample {i}")

        if i >= 4: # Run 5 samples maximum for the test
            break

    # Process User Uploaded Image if exists
    custom_image_path = "/jet/home/djariwal/.gemini/antigravity/brain/eac176ab-0a42-4319-9727-9cfcf8b46751/uploaded_image_1765471183335.png"
    if os.path.exists(custom_image_path):
        print(f"Processing custom image: {custom_image_path}")
        try:
            image = Image.open(custom_image_path).convert("RGB")
            # Default query for custom image
            query = "the main object"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Find {query}."},
                    ],
                }
            ]
            text_input = segmenter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = segmenter.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            att_map = segmenter.get_attention_map(inputs, image.size)
            if att_map is not None:
                overlay_img = apply_heatmap(image, att_map)
                overlay_img.save(os.path.join(output_dir, "custom_image_overlay.png"))
                print("Saved custom image result.")
        except Exception as e:
            print(f"Error processing custom image: {e}")

if __name__ == "__main__":
    main()
