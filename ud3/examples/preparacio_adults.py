#!/usr/bin/env python

from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder


def download(url, filepath):
    http_response = urlopen(url)
    content = http_response.read()
    with open(filepath, 'wb') as f:
        f.write(content)


files_folder = '../../files/ud3'
if not os.path.exists(files_folder):
    os.makedirs(files_folder)

file_path = os.path.join(files_folder, 'adult.data.csv')
if not os.path.exists(file_path):
    adult_dataset_url = 'https://raw.githubusercontent.com/joapuiib/saa-datasets/refs/heads/main/adult.data.csv'
    print('Downloading dataset...')
    download(adult_dataset_url, file_path)
else:
    print(f'Dataset found at {file_path}')


df = pd.read_csv(file_path)
print("Head of the dataset:")
print(df.head(), "\n")
print("Tail of the dataset:")
print(df.tail(), "\n")
print("Info of the dataset:")
df.info()
print()

print("Shape of the dataset:")
print(df.shape, "\n")
print("Description of the dataset:")
print(df.describe(), "\n")
print("Unique values of workclass column:")
print(df['workclass'].unique(), "\n")
print("Value counts of workclass column:")
print(df['workclass'].value_counts(dropna=False), "\n")


## Valors nuls
df.loc[50, 'age'] = None # Assignem un valor nul a la columna 'age' de la fila 50

print("Number of null values in each column:")
print(df.isnull().sum(), "\n")

## Eliminar files amb valors nuls
print(df[df['age'].isnull()])

rows = df.shape[0]
df.dropna(subset=['age'], inplace=True)
print(f"Deleted {rows - df.shape[0]} rows with null values in 'age' column", "\n")

print("Categorical columns:")
print(df.select_dtypes(include='object').columns, "\n")

# Assignem un valor '?' als valors nuls
# df.loc[df['workclass'].isnull(), 'workclass'] = '?'


workclass_oe = OrdinalEncoder()
df['workclass_oe'] = workclass_oe.fit_transform(df[['workclass']])

# print each unique value and its corresponding label
print("Unique values of workclass column:")
unique_pairs = df[['workclass', 'workclass_oe']].drop_duplicates().sort_values('workclass_oe').reset_index(drop=True)
print(unique_pairs, "\n")

encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['sex']])
encoded_columns = encoder.get_feature_names_out(['sex'])
df_encoded = pd.DataFrame(encoded_data.toarray(), columns=encoded_columns)
df = pd.concat([df, df_encoded], axis=1)

print("OneHot encoded columns:")
print(df[['sex', 'sex_Male', 'sex_Female']].head(), "\n")
