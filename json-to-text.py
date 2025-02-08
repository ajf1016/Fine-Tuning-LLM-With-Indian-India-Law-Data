import json
import pandas as pd

# Load IPC & CrPC JSON files
with open('dataset/ipc.json', 'r') as f:
    ipc_data = json.load(f)

with open('dataset/crpc.json', 'r') as f:
    crpc_data = json.load(f)

# Function to create dataset


def create_dataset(ipc, crpc):
    dataset = []
    for section in ipc['sections']:
        dataset.append({
            "prompt": f"Explain IPC Section {section['number']}",
            "completion": section['text']
        })
    for section in crpc['sections']:
        dataset.append({
            "prompt": f"Explain CrPC Section {section['number']}",
            "completion": section['text']
        })
    return dataset


# Create DataFrame and save as JSON
dataset = create_dataset(ipc_data, crpc_data)
df = pd.DataFrame(dataset)
df.to_json("legal_finetune_data.json", orient='records', indent=4)
