import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns


print("Scaricamento di SENTIPOLC in corso...")
dataset = load_dataset("evalitahf/sentiment_analysis")


df = pd.DataFrame(dataset['test'])

df['opos'] = pd.to_numeric(df['opos'], errors='coerce').fillna(0).astype(int)
df['oneg'] = pd.to_numeric(df['oneg'], errors='coerce').fillna(0).astype(int)
df = df[df['opos'] != df['oneg']].copy()

# 1. Dimensioni del dataset e tipi di dato
print("Dimensioni del dataset:", df.shape)
print("\nInfo sulle colonne:")
df.info()

# 2. Controllo dei valori nulli
print("\nValori mancanti per colonna:")
print(df.isnull().sum())

################################################

# Conteggio esatto classi
print(df['opos'].value_counts(normalize=True) * 100) # Mostra le percentuali

# Visualizzazione
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='opos', palette='viridis')
plt.title('Distribuzione del Sentiment (0 = Negativo, 1 = Positivo)')
plt.xlabel('Sentiment')
plt.ylabel('Numero di Tweet')
plt.show()

# Calcoliamo il numero di parole per ogni tweet
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

# Creiamo due grafici sovrapposti per vedere se la lunghezza varia in base al sentiment
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='word_count', hue='opos', bins=30, kde=True, palette='viridis')
plt.title('Distribuzione della lunghezza dei Tweet (in parole)')
plt.xlabel('Numero di parole')
plt.ylabel('Frequenza')
plt.show()



from collections import Counter

def get_top_n_words(corpus, n=20):
    # Unisce tutti i tweet, li divide in parole e le conta
    words = ' '.join(corpus.astype(str)).lower().split()
    return Counter(words).most_common(n)

# Parole più usate nei tweet negativi
top_words_neg = get_top_n_words(df[df['opos'] == 0]['text'])
# Parole più usate nei tweet positivi
top_words_pos = get_top_n_words(df[df['opos'] == 1]['text'])

print("Top 20 parole nei tweet NEGATIVI:\n", top_words_neg)
print("\nTop 20 parole nei tweet POSITIVI:\n", top_words_pos)