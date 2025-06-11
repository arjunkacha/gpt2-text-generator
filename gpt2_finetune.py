from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from transformers import pipeline

# Step 1: Load dataset from text file
dataset = load_dataset("text", data_files={"train": "sample_text_data.txt"})

# Step 2: Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Fix for padding

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 3: Tokenize the data
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=1,
    logging_steps=100,
    report_to="none"  # Disable wandb
)

# Step 5: Use data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 6: Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Step 7: Start Training
trainer.train()

# Step 8: Save the model
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

# Step 9: Generate text


generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="./gpt2-finetuned")

print("\n=== Generated Text ===")
output = generator("Once upon a time", max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])
