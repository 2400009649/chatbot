from datasets import load_dataset
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, TrainingArguments, Trainer

# Bước 1: Tải dữ liệu ViQuAD
dataset = load_dataset("NghiemAbe/viquad")

# Bước 2: Tiền xử lý dữ liệu
def preprocess_data(example):
    return {"input": example["prompt"], "response": example["answer"]}

processed_data = dataset["train"].map(preprocess_data)

# Chọn 1 tập nhỏ để tiết kiệm tài nguyên
train_data = processed_data.shuffle(seed=42).select(range(3000))
eval_data = processed_data.shuffle(seed=42).select(range(300))

# Bước 3: Tải mô hình và tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Tokenize dữ liệu
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input"], max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["response"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_eval = eval_data.map(tokenize_function, batched=True)

# Bước 4: Thiết lập Trainer
training_args = TrainingArguments(
    output_dir="./blenderbot_viquad",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
)

# Bước 5: Huấn luyện
trainer.train()

# Lưu mô hình
model.save_pretrained("./blenderbot_viquad")
tokenizer.save_pretrained("./blenderbot_viquad")
