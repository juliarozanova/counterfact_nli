from experiments.utils import encode
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_score(y_true=labels, y_pred=predictions)


if __name__=='__main__':
    dataset_name = 'snli'
    model_name = 'bert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


    train_dataset = load_dataset(dataset_name, split='train').filter(lambda x :  x['label']!=-1)
    train_dataset = train_dataset.map(lambda example: encode(example, tokenizer), batched=True)

    eval_dataset = load_dataset(dataset_name, split='validation').filter(lambda x :  x['label']!=-1)
    eval_dataset = eval_dataset.map(lambda example: encode(example, tokenizer), batched=True)


    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=True)