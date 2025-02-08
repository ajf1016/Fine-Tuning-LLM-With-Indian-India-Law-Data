import json
import pandas as pd
import os

# Define dataset directory
dataset_dir = "dataset"

# List of legal act JSON files
legal_files = ["ipc.json", "crpc.json", "hma.json", "ida.json",
               "iea.json", "cpc.json", "nia.json"]

# Function to load JSON data


def load_json(file_path):
    """Loads JSON and checks for errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not data or not isinstance(data, list):
                print(f"⚠️ WARNING: {file_path} has an invalid structure!")
                return None
            return data
    except Exception as e:
        print(f"❌ ERROR loading {file_path}: {e}")
        return None

# Function to create structured dataset


def create_dataset(legal_data):
    dataset = []
    for law_name, sections in legal_data.items():
        if sections:
            for section in sections:
                if "Section" in section and "section_desc" in section:
                    dataset.append({
                        "prompt": f"Explain {law_name.upper()} Section {section['Section']}: {section.get('section_title', 'No Title')}",
                        "completion": section['section_desc']
                    })
                else:
                    print(
                        f"⚠️ Skipping section in {law_name} due to missing keys: {section}")
    return dataset


# Load all legal act JSON files into a dictionary
legal_data = {}
for file in legal_files:
    file_path = os.path.join(dataset_dir, file)
    if os.path.exists(file_path):
        print(f"✅ Loading {file}...")
        legal_data[file.split('.')[0]] = load_json(file_path)
    else:
        print(f"❌ ERROR: {file} not found in {dataset_dir}")

# Create dataset
dataset = create_dataset(legal_data)

# Convert to DataFrame and save as JSON
df = pd.DataFrame(dataset)

if not df.empty:
    df.to_json("legal_finetune_data.json", orient='records',
               indent=4, force_ascii=False)
    print(f"🎉 Dataset created successfully with {len(df)} records!")
else:
    print("⚠️ No valid data found. Check the JSON files.")
