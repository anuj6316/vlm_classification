import torch
import os
import base64
import time
import yaml
from io import BytesIO
from PIL import Image
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    BitsAndBytesConfig
)
from olmocr.data.renderpdf import render_pdf_to_base64png
from pypdf import PdfReader

def process_pdf_smart(pdf_path):
    # 1. Configuration to prevent OOM
    # We use 4-bit quantization which fits the 7B model into ~5.5GB VRAM
    model_id = "allenai/olmOCR-2-7B-1025"
    print(f"Loading {model_id} in 4-bit mode...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # 2. Setup PDF and Prompt
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    latencies = []

    # Strict prompt for minimal output
    custom_prompt = (
        "Identify the document type (classification) and provide a concise 3-sentence summary. "
        "Output strictly as a YAML block with keys 'classification' and 'summary'. "
        "Do not include any other text."
    )

    print(f"\n--- ANALYZING {num_pages} PAGES ---")

    # 3. Processing Loop
    for i in range(1, num_pages + 1):
        try:
            start_time = time.perf_counter()

            # REDUCED RESOLUTION: 800px instead of 1288px to save memory
            image_base64 = render_pdf_to_base64png(pdf_path, i, target_longest_image_dim=800)
            image = Image.open(BytesIO(base64.b64decode(image_base64)))

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": custom_prompt}
                ]
            }]

            # Prepare Inputs
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to("cuda")

            # 4. Generate Output
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=250, 
                    temperature=0.1,
                    do_sample=False
                )
                
                # Slicing: Only decode NEW tokens (removes the system/user prompt)
                generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                result_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            latency = time.perf_counter() - start_time
            latencies.append(latency)

            # Clean output
            clean_result = result_text.strip().replace("```yaml", "").replace("```", "")
            print(f"[PAGE {i:02d}] {latency:.2f}s | {clean_result.splitlines()[0]}...") # Print first line of YAML

        except Exception as e:
            print(f"[PAGE {i:02d}] Error: {str(e)}")

    # 5. Final Report
    if latencies:
        avg_time = sum(latencies) / len(latencies)
        print("\n" + "="*40)
        print(f"AVERAGE TIME PER PAGE: {avg_time:.2f}s")
        print(f"TOTAL TIME FOR {num_pages} PAGES: {sum(latencies):.2f}s")
        print("="*40)

if __name__ == "__main__":
    PDF_FILE = "/content/TEST_K3-a.pdf"
    if os.path.exists(PDF_FILE):
        process_pdf_smart(PDF_FILE)
    else:
        print(f"File {PDF_FILE} not found.")