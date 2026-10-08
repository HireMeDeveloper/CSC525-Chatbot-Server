# Imports and dependencies
import json
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Constants
scriptDir = Path(__file__).parent
serverDir = scriptDir.parent
baseModel = "Qwen/Qwen3-0.6B"
dataDir = serverDir / "data"
conversationDatasets = sorted([str(filePath) for filePath in dataDir.glob("*.jsonl")])
chatbotOutput = str(serverDir / "model")
maxLength = 2048
rulesFile = str(serverDir / "rules.md")

# Train model function
def trainModel(model, tokenizer, dataset, outputDir, epochs=5, learningRate=2e-5, evaluate=True):
    dataset = dataset.train_test_split(test_size=0.1, seed=42) if evaluate else DatasetDict({"train": dataset})
    tokenized = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=maxLength), batched=True, remove_columns=dataset["train"].column_names)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=outputDir, num_train_epochs=epochs, per_device_train_batch_size=4, per_device_eval_batch_size=4, learning_rate=learningRate, weight_decay=0.01, logging_steps=10, eval_strategy="epoch" if evaluate else "no", save_strategy="epoch" if evaluate else "no", load_best_model_at_end=evaluate, save_total_limit=1, dataloader_num_workers=0, report_to="none"), train_dataset=tokenized["train"], eval_dataset=tokenized.get("test"), processing_class=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(outputDir)
    tokenizer.save_pretrained(outputDir)

# Load system prompt from rules.md
with open(rulesFile) as file:
    rules = file.read().strip()

# Load conversation datasets
conversationRecords = []
for datasetPath in conversationDatasets:
    with open(datasetPath) as file:
        for line in file:
            if cleanedLine := line.strip():
                try:
                    record = json.loads(cleanedLine)
                    conversationRecords.append(record)
                except json.JSONDecodeError:
                    continue
conversationDataset = Dataset.from_list(conversationRecords)

# Format conversation function
def formatConversation(example):
    messages = [{"role": "system", "content": rules}]
    if isinstance(example.get("messages"), list):
        for item in example["messages"]:
            messages.append({"role": item["role"].strip().lower(), "content": item["content"].strip()})
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except Exception:
        return {"text": ""}
    return {"text": formatted}

# Train the model
tokenizer = AutoTokenizer.from_pretrained(baseModel)
formattedConversations = conversationDataset.map(formatConversation).filter(lambda entry: isinstance(entry.get("text"), str) and len(entry["text"].strip()) > 0)
chatbotModel = AutoModelForCausalLM.from_pretrained(baseModel, dtype="auto")
trainModel(model=chatbotModel, tokenizer=tokenizer, dataset=formattedConversations, outputDir=chatbotOutput, epochs=1, learningRate=2e-5, evaluate=False)
print("✓ Training complete")
