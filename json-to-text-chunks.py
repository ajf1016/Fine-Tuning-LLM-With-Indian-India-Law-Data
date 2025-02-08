import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load IPC JSON (Example)
with open("legal_finetune_data.json", "r", encoding="utf-8") as f:
    ipc_data = json.load(f)

# Extract Sections & Convert to Text Format
text_data = []
for section in ipc_data:
    text_data.append(
        f"IPC Section {section['Section']}: {section['section_desc']}")

# Split Text into Chunks (RAG-friendly)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50)
documents = text_splitter.create_documents(text_data)

print(f"✅ {len(documents)} chunks created!")
