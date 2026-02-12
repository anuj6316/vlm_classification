import torch
import os
import base64
import time
import csv
from io import BytesIO
from PIL import Image
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    BitsAndBytesConfig
)
from olmocr.data.renderpdf import render_pdf_to_base64png
from pypdf import PdfReader

def process_and_save_results(pdf_path, output_csv="results.csv"):
    # 1. Load Model in 4-bit
    model_id = "./olmocr_model"
    print(f"Loading {model_id}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    
    # Prepare CSV header
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Page", "Time (s)", "Classification", "Summary"])

    custom_prompt = (
        "Identify the document type (classification) and provide a concise 3-sentence summary. "
        "Output strictly as a YAML block with keys 'classification' and 'summary'."
    )

    print(f"\n--- PROCESSING {num_pages} PAGES ---")

    for i in range(1, num_pages + 1):
        try:
            start_time = time.perf_counter()

            # Render page (800px for speed/VRAM balance)
            image_base64 = render_pdf_to_base64png(pdf_path, i, target_longest_image_dim=800)
            image = Image.open(BytesIO(base64.b64decode(image_base64)))

            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": custom_prompt}]}]
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to("cuda")

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=200, 
                    do_sample=False  # Removed temperature to fix warning
                )
                
                generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                result_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            latency = time.perf_counter() - start_time
            
            # Simple parsing for display and CSV
            # This assumes the model outputs YAML as requested
            lines = result_text.strip().replace("```yaml", "").replace("```", "").splitlines()
            classification = "Unknown"
            summary = "No summary generated"
            
            for line in lines:
                if "classification:" in line.lower():
                    classification = line.split(":", 1)[1].strip()
                elif "summary:" in line.lower():
                    summary = line.split(":", 1)[1].strip()

            # Print to console
            print(f"[PAGE {i:02d}] {latency:.2f}s | CLASS: {classification} | SUM: {summary[:60]}...")

            # Save to CSV
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i, f"{latency:.2f}", classification, summary])

        except Exception as e:
            print(f"[PAGE {i:02d}] Error: {str(e)}")

    print(f"\n✅ Done! Results saved to {output_csv}")

if __name__ == "__main__":
    process_and_save_results("/home/mindmap/vlm_classification/input/TEST_K3.pdf")