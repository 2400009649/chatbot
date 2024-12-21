from deep_translator import GoogleTranslator
from datasets import load_dataset

# Tải dữ liệu
dataset = load_dataset("daily_dialog", split="train")

def translate_dialog(dialogs):
    translator = GoogleTranslator(source="en", target="vi")
    translated = [translator.translate(dialog) for dialog in dialogs]
    return translated

# Dịch dữ liệu
dataset = dataset.map(lambda x: {"dialog_vi": translate_dialog(x["dialog"])})
dataset.to_json("daily_dialog_vi.json")
