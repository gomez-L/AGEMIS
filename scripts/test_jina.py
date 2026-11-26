import torch
from transformers import AutoModel, AutoTokenizer
model_id = "jinaai/jina-embeddings-v5-text-nano"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    inputs = tokenizer("test", return_tensors="pt")
    outputs = model(**inputs)
    print("Success! Output shape:", outputs[0].shape)
except Exception as e:
    import traceback
    traceback.print_exc()
