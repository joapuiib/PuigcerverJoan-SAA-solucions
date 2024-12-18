import pandas as pd

from sklearn.model_selection import train_test_split


df = pd.read_csv('../../files/ud3/adult.data.csv')

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = 0.7
validation_size = 0.15
test_size = 0.15

train_end = int(len(df_shuffled) * train_size)
validation_end = train_end + int(len(df_shuffled) * validation_size)

p_train_df = df_shuffled.iloc[:train_end]
p_validation_df = df_shuffled.iloc[train_end:validation_end]
p_test_df = df_shuffled.iloc[validation_end:]

print(f'Original data set: {df.shape}')
print()

print("Pandas")
print(f'Train set: {p_train_df.shape}')
print(f'Validation set: {p_validation_df.shape}')
print(f'Test set: {p_test_df.shape}')
print()


sk_train_df, val_test_df = train_test_split(df, test_size=0.3, random_state=42)
sk_validation_df, sk_test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

print("Scikit-learn")
print(f'Train set: {sk_train_df.shape}')
print(f'Validation set: {sk_validation_df.shape}')
print(f'Test set: {sk_test_df.shape}')