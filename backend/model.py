from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import PromptTuningConfig, get_peft_model, PromptTuningInit
from datasets import load_dataset

# Load base model and tokenizer
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the prompt tuning configuration
peft_config = PromptTuningConfig(
    task_type="SEQ_2_SEQ_LM",  # Task type (e.g., sequence-to-sequence for summarization)
    num_virtual_tokens=20,    # Number of virtual tokens to prepend
    prompt_tuning_init=PromptTuningInit.TEXT,  # Initialize prompt embeddings from text
    prompt_tuning_init_text="Summarize this:",  # Initialization text for the prompt
    tokenizer_name_or_path=model_name,  # Specify the tokenizer name or path
)

# Wrap the model with PEFT
peft_model = get_peft_model(model, peft_config)

# Load a summarization dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    targets = examples["highlights"]
    
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    predict_with_generate=True,
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save prompt-tuned model
peft_model.save_pretrained("./prompt_tuned_model")