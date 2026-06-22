import os
import tensorflow as tf
import tensorflow_datasets as tfds

from src.logger import logging

class DataIngestion:
    def __init__(self, local_dir="./data", dataset_name="tiny_shakespeare", batch_size=32, max_len=64):
        self.local_dir = local_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.file_path = os.path.join(self.local_dir, f"{self.dataset_name}.txt")
        
        if not os.path.exists(self.local_dir):
            logging.info(f"Creating local data directory at {self.local_dir}")
            os.makedirs(self.local_dir)

    def load_data(self):
        if os.path.exists(self.file_path):
            logging.info(f"Loading dataset locally from: {self.file_path}")
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        else:
            logging.info(f"Local file not found. Fetching '{self.dataset_name}' from TensorFlow Datasets")
            ds = tfds.load(self.dataset_name, split='train')
            for example in ds.take(1):
                raw_text = example['text'].numpy().decode('utf-8')
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            logging.info(f"Saved dataset locally to {self.file_path}")
            
        return raw_text

    def prepare_pipeline(self, raw_text, tokenizer):
        vectorized_text = tokenizer([raw_text])[0]
        
        input_sequences = []
        target_sequences = []
        
        num_samples = len(vectorized_text) - self.max_len
    
        for i in range(0, num_samples - 1, self.max_len):
            input_sequences.append(vectorized_text[i : i + self.max_len])
            target_sequences.append(vectorized_text[i + 1 : i + self.max_len + 1])
            
        dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))
        
        logging.info(f"Created TensorFlow dataset with {len(input_sequences)} samples")
        dataset = (dataset
                   .shuffle(1000)
                   .batch(self.batch_size)
                   .cache()
                   .prefetch(tf.data.AUTOTUNE))
        return dataset