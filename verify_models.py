import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor, ViTModel
from PIL import Image
import warnings

# Suppress some common huggingface warnings
warnings.filterwarnings("ignore")

def verify_models(image_path):
    print("Loading image...")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully. Size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print("\n--- 1. Testing ConvNeXt Model ---")
    convnext_model_id = "DrGM/DrGM-ConvNeXt-V2L-Facial-Emotion-Recognition"
    try:
        print(f"Loading processor and model for {convnext_model_id}...")
        conv_processor = AutoImageProcessor.from_pretrained(convnext_model_id)
        conv_model = AutoModelForImageClassification.from_pretrained(convnext_model_id)
        
        print("Processing image...")
        inputs = conv_processor(images=image, return_tensors="pt")
        
        print("Running forward pass...")
        with torch.no_grad():
            outputs = conv_model(**inputs)
            
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        print(f"ConvNeXt output logits shape: {logits.shape}")
        print(f"Predicted emotion class index: {predicted_class_idx}")
        if conv_model.config.id2label:
            print(f"Predicted emotion label: {conv_model.config.id2label[predicted_class_idx]}")
            
    except Exception as e:
        print(f"Error with ConvNeXt: {e}")

    print("\n--- 2. Testing DINO Vision Transformer ---")
    dino_model_id = "facebook/dino-vits16"
    try:
        print(f"Loading processor and model for {dino_model_id}...")
        # DINO models use ViT processing
        dino_processor = ViTImageProcessor.from_pretrained(dino_model_id)
        dino_model = ViTModel.from_pretrained(dino_model_id)
        
        print("Processing image...")
        inputs = dino_processor(images=image, return_tensors="pt")
        
        print("Running forward pass...")
        with torch.no_grad():
            outputs = dino_model(**inputs)
            
        last_hidden_states = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        
        print(f"DINO last hidden state shape (batch_size, sequence_length, hidden_size): {last_hidden_states.shape}")
        if pooler_output is not None:
            print(f"DINO pooler output shape (CLS token representation): {pooler_output.shape}")
            
        print("DINO feature extraction successful!")
        
    except Exception as e:
        print(f"Error with DINO: {e}")

if __name__ == "__main__":
    test_img = "DATASET/test/1/test_0002_aligned.jpg"
    print(f"Starting verification with image: {test_img}")
    verify_models(test_img)
