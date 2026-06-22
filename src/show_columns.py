from src.data_loader import load_data

df = load_data()

print("\nCOLUNAS DO DATASET:\n")

for col in df.columns:
    print(col)
    