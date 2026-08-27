#   Christian Lemma
#   CMSC437
#   5/06/2025
#
#
# Title: Word Sense Disambiguation Scorer
#
#
# Description:
#
#   This program evaluates the performance of the word sense disambiguation program (wsd-embeddings.py) by
#   comparing its predicted senses against a key-file that satisfies all expectations. It calculates the 
#   overall accuracy and generates a confusion matrix to analyze classification performance.
#
#
#
# Algorithm Used:
#
#   1. Reading and parsing through data
#       - Reads the key file and prediction file
#       - Extracts instance IDs and the corresponding sense label for each ID
#
#   2. Evaluation the predictions
#       - Each predicted sense is compared with the actual sense from the key file
#       - Creates a confusion matrix that tracks correct and incorrect classification
#       - The overall overage accuracy gets calculated.
#
#   3. Output
#       - Prints the overall accuracy
#       - The confusion matrix is displayed which shows the classification performance for each sense.
#
#
# User Guide:
#
#   Run this program from the command line using the following commands:
#       -python3 scorer.py [prediction file] [key file]
#
#   -[prediciton file] : file containing the model's classified outputs
#   -[key file] : file containing the correct sense labels
#
#
# Sample Outputs:
#
# Overall Accuracy: 83.33%
#                                                                                                       
# Confusion Matrix:
#        phone   product
# -----------------------
# phone   59      13
# product 8       46
#
#
#

import sys
from collections import defaultdict
import re

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding= "utf-16") as f:
            return f.read()
    except:
        with open(file_path, 'r') as f:
            return f.read()

def parse_sense(text):
    # extract instance ID and senseid
    info = {}
    
    #split file content at each <answer tag to mark individual instances
    blocks = re.findall(r'<answer\s+instance="([^"]+)"\s+senseid="([^"]+)"\s*/?>', text)
    for instance_id, sense in blocks:
        # store in dictionary
        info[instance_id]=sense
    return info


def evaluate(predictions, key):
    # comparing predicted senses with key file and compute evaluation
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0

    # iterate over each predicted instance and compare against key file
    for instance_id, correct_sense in predictions.items():
        true_sense = key.get(instance_id, None)
        if true_sense is None:
            continue
        if correct_sense == true_sense:
            correct += 1
        confusion_matrix[true_sense][correct_sense] += 1
        total += 1
    
    return confusion_matrix, correct, total



def print_confusion_matrix(confusion_matrix, correct, total):
    
    # print confusion matrix and accuracy percentage
    accuracy = correct / total * 100 if total else 0
    print(f"Overall Accuracy: {accuracy:.2f}%\n")

    labels = sorted(set(confusion_matrix.keys()) | {sense for row in confusion_matrix.values() for sense in row})
    print("\nConfusion Matrix:")
    header = "\t" + "\t".join(labels)
    print(header)
    print("-" * len(header.expandtabs()))
    for actual in labels:
        row = [str(confusion_matrix[actual].get(predicted, 0)) for predicted in labels]
        print(f"{actual}\t" + "\t".join(row))

def main():
    # 
    if len(sys.argv) != 3:
        print("Usage: python3 scorer.py <predictions-file> <key-file>")
        sys.exit(1)
    
    predictions_text = read_file(sys.argv[1])
    key_text = read_file(sys.argv[2])

    predictions = parse_sense(predictions_text)
    key = parse_sense(key_text)

    confusion_matrix, correct, total = evaluate(predictions, key)
    print_confusion_matrix(confusion_matrix, correct, total)


if __name__ == '__main__':
    main()