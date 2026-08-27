import sys
import re
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim

# This checks if the GPU is ready for pytorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This function loads the GloVe embeddings into a dictionary
def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf8') as f: #read given glove txt file
        for line in f:
            parts = line.strip().split()
            word = parts[0]                              # the first token is the word
            vec = np.array(parts[1:], dtype=float)       # the remaining tokens are the vector
            embeddings[word] = vec
    return embeddings

# this function removes the XML tags and other punctuation, then returns lowercase tokens
def clean_text(text):
    text = re.sub(r'<head>(.*?)</head>', r'\1', text)   # strip <head> tags
    text = re.sub(r'<[^>]+>', '', text)                 # remove any other html tags
    return re.sub(r'[^\w\s]', '', text).lower().split() # remove punctuation and lowercase

# This function calculates average GloVe vector for all identified words in the sentence
#       - This is done by adding the individual vector of each word and 
#         dividing by the total number of words in the sentence.
def get_context_vector(tokens, embeddings, dim=100):
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    if not vectors:
        return np.zeros(dim) # for no identified words
    return np.mean(vectors, axis=0)

# This function will parse training data and extract feature vectors, the labels, and the instance ids
def parse_train(file_path, embeddings, dim=100):
    X, Y, ids = [], [], []
    with open(file_path, 'r') as f:
        data = re.split(r'(?=<instance id=)', f.read()) # this splits each instance block
        for block in data:
            match = re.search(r'<answer instance="([^"]+)" senseid="([^"]+)"/>', block)
            context_match = re.search(r'<s>(.*?)</s>', block, re.DOTALL)
            if not match or not context_match:
                continue # skip incomplete blocks
            instance_id = match.group(1)
            label = match.group(2)
            context = clean_text(context_match.group(1)) # tokenized context
            vec = get_context_vector(context, embeddings, dim)
            X.append(vec)
            Y.append(0 if label == "phone" else 1)
            ids.append(instance_id)
    return np.array(X), np.array(Y), ids

# this function will parse the test data only by feature vectors and the instance ids
def parse_test(file_path, embeddings, dim=100):
    X, ids = [], []
    with open(file_path, 'r') as f:
        data = re.split(r'(?=<instance id=)', f.read())
        for block in data:
            match = re.search(r'<instance id="([^"]+)">', block)
            context_match = re.search(r'<s>(.*?)</s>', block, re.DOTALL)
            if not match or not context_match:
                continue
            instance_id = match.group(1)
            context = clean_text(context_match.group(1))
            vec = get_context_vector(context, embeddings, dim)
            X.append(vec)
            ids.append(instance_id)
    return np.array(X), ids

# This is the neural network definition for classification
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), # first layer
            nn.ReLU(),                # activation
            nn.Linear(64, 2)          # output for 2 classes
        )

    def forward(self, x):
        return self.fc(x)

# This function trains the NN model on the training data and returns predictions on the test set
def predict_with_nn(X_train, y_train, X_test):
    model = SimpleNN(X_train.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # convert array to pytorch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

    # 20 epoch training
    for _ in range(20):  # small epoch count for quick training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    #evaluate on test set
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy() # get class with highest score
    return preds

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 wsd-embeddings.py line-train.txt line-test.txt glove.txt [SVM|NN]")
        sys.exit(1)

    # parse command line args
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    glove_file = sys.argv[3]
    model_type = sys.argv[4] if len(sys.argv) > 4 else "SVM"
    
    # embedding demension from file name
    dim = 100 if "100d" in glove_file else 50  # crude dimension detection
    
    # load embedding and datasets
    embeddings = load_glove_embeddings(glove_file)
    X_train, y_train, _ = parse_train(train_file, embeddings, dim)
    X_test, test_ids = parse_test(test_file, embeddings, dim)

    # choose classifier
    if model_type.upper() == "NN":
        predictions = predict_with_nn(X_train, y_train, X_test)
    else:
        model = SVC(kernel='linear') # the default is support vector machine
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    # output answer in <answer> format
    for instance_id, pred in zip(test_ids, predictions):
        label = "phone" if pred == 0 else "product"
        print(f'<answer instance="{instance_id}" senseid="{label}"/>')

if __name__ == "__main__":
    main()
