"""Check if SeeClick model exists on HuggingFace"""
from huggingface_hub import model_info

models_to_check = [
    "cckevinn/SeeClick",
    "SeeClick/SeeClick", 
    "microsoft/SeeClick",
    "OmniParser/icon-detect",
    "microsoft/Florence-2-large",
    "Salesforce/blip2-opt-2.7b"
]

print("Checking which models exist on HuggingFace:\n")
for model_name in models_to_check:
    try:
        info = model_info(model_name)
        print(f"✅ {model_name} - EXISTS")
        print(f"   Downloads: {info.downloads}")
        print(f"   Tags: {info.tags[:5] if info.tags else 'None'}\n")
    except Exception as e:
        print(f"❌ {model_name} - NOT FOUND")
        print(f"   Error: {str(e)[:60]}...\n")
