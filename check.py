with open('train_sentiments.py', 'r') as f:
    content = f.read()

checks = {
    'max_features=50000': 'max_features=50000' in content,
    'ngram_range=(1, 2)': 'ngram_range=(1, 2)' in content,
    'C=5.0': 'C=5.0' in content,
    'apply(clean_text)': 'apply(clean_text)' in content,
}

for k, v in checks.items():
    status = 'FOUND' if v else 'MISSING'
    print(f"{status} --> {k}")