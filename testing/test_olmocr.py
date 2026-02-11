import torch
import os
import base64
import time
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from olmocr.data.renderpdf import render_pdf_to_base64png
from pypdf import PdfReader

def process_with_timing(pdf_path):
    # 1. Load Model (Using FP8 for optimal speed)
    model_id = "allenai/olmOCR-2-7B-1025"
    print(f"Loading {model_id}...")
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quantization_config, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # 2. Setup PDF and Timing Metrics
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    latencies = []

    # Custom prompt for strictly Summary and Classification
    custom_prompt = """
    Identify the document type (classification) and provide a concise 3-sentence summary.
    Output your response strictly as a YAML block:
    - classification: [Type]
    - summary: [Summary Text]
    """

    print(f"\n--- ANALYZING {num_pages} PAGES ---")

    # 3. Processing Loop with Latency Tracking
    for i in range(1, num_pages + 1):
        # Start timer for this page
        start_time = time.perf_counter()

        # Render and Process
        image_base64 = render_pdf_to_base64png(pdf_path, i, target_longest_image_dim=1288)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": custom_prompt}
        ]}]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
        
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Stop timer
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)

        print(f"[PAGE {i:02d}] Time: {latency:.2f}s | Result: {generated_text.strip()[:50]}...")

    # 4. Final Performance Report
    if latencies:
        avg_time = sum(latencies) / len(latencies)
        print("\n" + "="*30)
        print("PERFORMANCE SUMMARY")
        print("="*30)
        print(f"Total Pages Processed: {len(latencies)}")
        print(f"Average Time per Page: {avg_time:.2f} seconds")
        print(f"Total Processing Time: {sum(latencies):.2f} seconds")
        print("="*30)

if __name__ == "__main__":
    process_with_timing("/content/TEST_K3-a.pdf")