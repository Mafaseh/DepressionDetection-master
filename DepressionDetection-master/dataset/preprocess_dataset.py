import pandas as pd
from collections import Counter
from utils import get_data, labels  # Don't worry about this error just run this class anyways
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

dev_probability = {
    'severe': 0.09,
    'moderate': 0.51,
    'not depression': 0.4
}

def preprocess(with_test=False):
    train = get_data('train')
    statistics('train', train)

    dev = get_data('dev')
    statistics('dev', dev)

    buckets = {l: [] for l in labels}
    all_texts = set()
    for data in [train, dev]:
        for idx, row in data.iterrows():
            pid = row['pid']
            text = row['text']
            label = row['labels']
            if text not in all_texts:
                buckets[labels[label]].append([pid, text, label])
                all_texts.add(text)

    train_dataset, dev_dataset = [], []
    for label, prob in dev_probability.items():
        v = int(prob * 1000)
        train_dataset += buckets[label][:-v]
        dev_dataset += buckets[label][-v:]

    train = pd.DataFrame(train_dataset, columns=['pid', 'text', 'labels'])
    print_stats('Train after preprocessing', train)
    train.to_csv('../data/preprocessed_dataset/train.csv', index=False)

    dev = pd.DataFrame(dev_dataset, columns=['pid', 'text', 'labels'])
    print_stats('Dev after preprocessing', dev)
    dev.to_csv('../data/preprocessed_dataset/dev.csv', index=False)

    if with_test:
        test = get_data('test')
        statistics('test', test)
        test_features = vectorizer.transform(test['text'])
        test_labels = test['labels']

        # Preprocess test dataset
        test_dataset = []
        for idx, row in test.iterrows():
            pid = row['pid']
            text = row['text']
            label = row['labels']
            test_dataset.append([pid, text, label])

        test = pd.DataFrame(test_dataset, columns=['pid', 'text', 'labels'])
        print_stats('Test after preprocessing', test)
        test.to_csv('../data/preprocessed_dataset/test.csv', index=False)

        calculate_accuracy(train_features, train_labels, dev_features, dev_labels, test_features, test_labels)


def statistics(data_split, dataset):
    unique_data = []
    all_texts = set()
    for idx, row in dataset.iterrows():
        if row['text'].lower() not in all_texts:
            unique_data.append([row['pid'], row['text'], row['labels']])
            all_texts.add(row['text'].lower())

    print_stats(f'Original {data_split}', dataset)

    df = pd.DataFrame(unique_data, columns=['pid', 'text', 'labels'])
    print_stats(f'Original {data_split} - without duplicates', df)


def print_stats(description, dataset):
    print(description)
    counts = Counter(list(dataset['labels'].values))
    for idx, label in enumerate(labels):
        print(f'{label}: {counts[idx]}')
    print(f'all: {len(dataset)}')
    print('-------------------------------------')


def calculate_accuracy(train_features, train_labels, dev_features, dev_labels, test_features, test_labels):
    combined_features = np.concatenate((train_features.toarray(), dev_features.toarray()), axis=0)
    combined_labels = np.concatenate((train_labels, dev_labels), axis=0)

    # Split the combined dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(combined_features, combined_labels,
                                                                        test_size=0.2, random_state=42)

    # Perform oversampling using SMOTE to address class imbalance
    oversampler = SMOTE(random_state=42)
    train_data, train_labels = oversampler.fit_resample(train_data, train_labels)

    # Train a Logistic Regression classifier
    model = LogisticRegression(random_state=42, max_iter=1000)  # Set max_iter to 1000 for convergence
    model.fit(train_data, train_labels)

    # Make predictions on the test data
    predictions = model.predict(test_data)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    train = get_data('train')
    statistics('train', train)

    dev = get_data('dev')
    statistics('dev', dev)

    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train['text'])
    train_labels = train['labels']

    dev_features = vectorizer.transform(dev['text'])
    dev_labels = dev['labels']

    preprocess(with_test=True)
