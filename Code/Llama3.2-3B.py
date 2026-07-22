#This file of code is executed in Google Colab

!pip install -q huggingface_hub
from huggingface_hub import login
login()
!pip install "transformers>=5.2.0"

import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

import pandas as pd
import re
import os
import csv
from google.colab import drive

try:
   drive.mount('/content/drive')
except ValueError:
   drive.mount('/content/drive', force_remount=True)


BASE_PATH = "" # path removed
INPUT_CSV = os.path.join(BASE_PATH, "iSarcasmEval-1400.csv")
df_input = pd.read_csv(INPUT_CSV)
sentences = df_input["Sentence"].astype(str).tolist()
print(f"Loaded {len(sentences)} sentences from CSV.")

model_name = "Llama3.2-3b"
test = "zero" # Replace with "zero+conservative" and "COT" and "COT+conservative"
filename = model_name + "_" + test + "_iSarcasmEval.csv"
print(filename)
filepath = BASE_PATH + model_name + "/"
print(filepath)

# Paths
BASE_PATH = filepath
raw_output_path = os.path.join(BASE_PATH, "Raw_" + filename)
table_output_path = os.path.join(BASE_PATH, "Table_" + filename)

# Load already processed sentences (if file exists)
processed_sentences = set()
if os.path.exists(table_output_path):
    with open(table_output_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_sentences.add(row["Sentence"])

print(f"Found {len(processed_sentences)} previously processed sentences.")

# Open files in append mode
raw_file = open(raw_output_path, mode="a", newline="", encoding="utf-8-sig")
table_file = open(table_output_path, mode="a", newline="", encoding="utf-8-sig")

raw_writer = csv.writer(raw_file)
table_writer = csv.writer(table_file)

# If files are new, write headers
if os.path.getsize(raw_output_path) == 0:
    raw_writer.writerow(["Line"])
if os.path.getsize(table_output_path) == 0:
    table_writer.writerow(["Sentence", "Sarcasm Polarity", "Intensity Score"])

# Process sentences
for sentence in sentences:
    if sentence in processed_sentences:
        print(f"Skipping already processed: {sentence}")
        continue

    prompt = f"""Instruction: Determine whether the following input text expresses sarcasm, if it does, output 'sarcastic', otherwise, output 'non-sarcastic'.

Input: {sentence}

Output:"""

    reply = pipe(
        prompt,
        max_new_tokens=2000,
        temperature=1.0,
        do_sample=True,
    )

    reply_text = reply[0]["generated_text"]

    match = re.search(
        r"Output:\s*['\"]?(sarcastic|non-sarcastic)['\"]?",
        reply_text,
        re.IGNORECASE
    )

    polarity = match.group(1) if match else "Unknown"

    score_matches = re.findall(r"-?\d+", reply_text)
    score = int(score_matches[-1]) if score_matches else None

    # Save raw output immediately
    raw_writer.writerow([sentence])
    raw_writer.writerow([reply])
    raw_writer.writerow(["-"*20])
    raw_file.flush()

    # Save structured table row immediately
    table_writer.writerow([sentence, polarity, score])
    table_file.flush()

    print(f"Processed and saved: {sentence}")

raw_file.close()
table_file.close()
