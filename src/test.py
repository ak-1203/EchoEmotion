from data_loader import load_dataset

X, y = load_dataset("../data/ravdess")

print(X.shape)
print(y[:10])