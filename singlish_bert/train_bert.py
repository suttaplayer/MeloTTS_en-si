from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import pandas as pd
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
DATA_PATH = "./data/sinhala-pali_sentences-bert.csv"
pandas_df = pd.read_csv(DATA_PATH, sep='|')
ds = Dataset.from_pandas(pandas_df)
train_test_dataset = ds.train_test_split()

# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(train_dataset):
    inputs = tokenizer(train_dataset['Sinhala_Sentences'], padding='max_length', truncation=True)
    inputs['labels'] = tokenizer(train_dataset['Sinhala_Sentences'], padding='max_length', truncation=True)['input_ids']
    return inputs

tokenized_dataset = train_test_dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_dataset['train']
test_dataset = tokenized_dataset['test']

# Define training arguments
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Load model
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

# Move model to device
model.to(device)

# Define compute_metrics function
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions.argmax(-1)
    return {"accuracy": (pred_ids == labels_ids).mean()}

# Define training step
def compute_loss(model, inputs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, model.config.vocab_size), labels.view(-1))
    return loss

# Define Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, model.config.vocab_size), labels.view(-1))
        return loss

# Initialize trainer with custom Trainer class
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save the trained model
output_model_dir = './model'
trainer.save_model(output_model_dir)