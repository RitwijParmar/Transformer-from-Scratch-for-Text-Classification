import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# Import project modules
import data_setup
import model as model_arch
import engine
import utils

# --- CONFIGURATION ---
DATA_FILEPATH = 'Tweets.csv'
EMBED_DIM = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
NUM_CLASSES = 3
DROPOUT = 0.2
EPOCHS = 10
BATCH_SIZE = 128
MAX_LEN = 40
LEARNING_RATE = 0.001

# --- SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = data_setup.load_and_preprocess_data(DATA_FILEPATH)
train_loader, val_loader, test_loader, vocab = data_setup.create_dataloaders(df, BATCH_SIZE, MAX_LEN)

# --- MODEL ---
VOCAB_SIZE = len(vocab)
model = model_arch.TransformerClassifier(
    VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_CLASSES, DROPOUT
).to(device)
print("\nModel Architecture:")
summary(model)

# --- TRAINING ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
criterion = nn.CrossEntropyLoss().to(device)

print("\n--- Starting Model Training ---")
history = engine.train_model(
    model, train_loader, val_loader, optimizer, scheduler, criterion, EPOCHS, device
)

# --- EVALUATION ---
print("\n--- Evaluating on Test Set ---")
utils.evaluate_and_report(model, test_loader, device, NUM_CLASSES)

print("\n--- Plotting Training Curves ---")
utils.plot_training_curves(history, "Transformer Model Training")