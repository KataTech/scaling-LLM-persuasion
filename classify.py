import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# ---------------------------
# Load HuggingFace zero-shot classifier
# ---------------------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ---------------------------
# Main pipeline
# ---------------------------
def run_classification(df, text_column="text", prompts_dir="prompts", threshold=0.5):
    # Load all prompt files
    prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith(".txt")]
    labels = [pf.split("_")[0] for pf in prompt_files]  # trait names
    instructions = [open(os.path.join(prompts_dir, pf)).read().strip() for pf in prompt_files]
    print("Loaded traits:", labels)

    # For each message, classify against all instructions at once
    results = []
    for msg in tqdm(df[text_column],  desc="Classifying messages"):
        res = classifier(msg, candidate_labels=instructions, multi_label=True)
        # Map back to trait prefixes
        row = {labels[i]: int(res["scores"][i] >= threshold) for i in range(len(labels))}
        results.append(row)
    # Append new columns to dataframe
    for trait in labels:
        df[trait] = [r[trait] for r in results]

    return df

if __name__ == "__main__":
    df = pd.read_csv("llm_responses.csv")
    df = run_classification(df, text_column="response", prompts_dir="prompts", threshold=0.5)
    df.to_csv("llm_responses_labeled.csv", index=False)
    print(df.head())
