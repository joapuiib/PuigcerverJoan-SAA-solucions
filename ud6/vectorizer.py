from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.DataFrame({
    'text': [
        'El perro corre rápido',
        'El gato corre lento',
        'El perro y el gato son amigos'
    ]
})

vectorizer = CountVectorizer(min_df=1, strip_accents='unicode', token_pattern=r'\w{3,}', ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df['text'])

# Convert sparse matrix to DataFrame with feature names as column headers
df_features = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(df_features)
print(df_features.columns)