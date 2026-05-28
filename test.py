import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import re
import spacy

print("Caricamento modello linguistico spaCy in corso...")

nlp = spacy.load("it_core_news_sm")

# spaCy "non" non fra le stopwords
nlp.vocab["non"].is_stop = False


# 1) Funzioni di Pre-Processing

def pulisci_e_lemmatizza(testo):
    # Pulizia tramite Regex
    testo = re.sub(r'http\S+|www\S+|<URL>', '', testo)
    testo = re.sub(r'@\w+|<MENTION_\d+>', '', testo)
    testo = re.sub(r'#', '', testo)
    testo = re.sub(r'[^a-zA-Zàèéìòù\s]', ' ', testo)
    testo = testo.lower()
    testo = re.sub(r'\s+', ' ', testo).strip()

    if not testo:
        return ""

    # Lemmatizzazione con spaCy
    doc = nlp(testo)
    lemmi = []
    for token in doc:
        if not token.is_stop and len(token.text) > 1:
            lemmi.append(token.lemma_)

    return " ".join(lemmi)


def pulizia_sentipolc(hf_dataset_split):
    """Rimuove tweet con sentiment neutro e misto, creando un target binario"""
    df = pd.DataFrame(hf_dataset_split)

    """Controllare la presenza di NaN in opos/oneg, poi controllare l'altra colonna; se non determinabile, .dropna()"""

    df['opos'] = pd.to_numeric(df['opos'], errors='coerce').fillna(0).astype(int)
    df['oneg'] = pd.to_numeric(df['oneg'], errors='coerce').fillna(0).astype(int)

    ###########################################################################################################
    df_filtered = df[df['opos'] != df['oneg']].copy()
    df_filtered['label'] = df_filtered['opos']
    return df_filtered[['text', 'label']]


# 2) Caricamento e Preparazione Dati


print("Scaricamento di SENTIPOLC in corso...")
dataset = load_dataset("evalitahf/sentiment_analysis")

df_train_full = pulizia_sentipolc(dataset['train'])
df_test_official = pulizia_sentipolc(dataset['test'])

print("Lemmatizzazione Train Set...")
df_train_full['text'] = df_train_full['text'].apply(pulisci_e_lemmatizza)

print("Lemmatizzazione Test Set...")
df_test_official['text'] = df_test_official['text'].apply(pulisci_e_lemmatizza)

print(f"Righe Train Set: {len(df_train_full)}")
print(f"Righe Test Set: {len(df_test_official)}")

X_train_full = df_train_full['text']
y_train_full = df_train_full['label']

X_test_official = df_test_official['text']
y_test_official = df_test_official['label']


# 3) Grid search e definizione Pipeline


print("\n" + "=" * 70)
print(" Ricerca dei parametri ottimali tramite Grid Search")
print("=" * 70)

# Costruzione della Pipeline (Vettorizzatore + Modello [balanced])
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# Definizione della griglia dei parametri
griglia_parametri = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [2, 3, 5],
    'tfidf__max_df': [0.75, 0.85, 0.95],
    'clf__C': [0.1, 1, 10]
}

# Grid Search, massimizzando il f1_macro
grid_search = GridSearchCV(
    pipeline_lr,
    griglia_parametri,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

# Addestramento sul Train Set
grid_search.fit(X_train_full, y_train_full)


# 4) Risultati


print("\n--- RISULTATI GRID SEARCH ---")
print(f"Miglior Macro F1-score in validazione interna: {grid_search.best_score_:.4f}")
print("I parametri perfetti trovati sono:")
for param_name in sorted(griglia_parametri.keys()):
    print(f" - {param_name}: {grid_search.best_params_[param_name]}")

# Valutazione finale sul Test Set Ufficiale
modello_vincitore = grid_search.best_estimator_
predizioni_test = modello_vincitore.predict(X_test_official)

print("\n--- REPORT SUL TEST SET FINALE ---")
print(classification_report(y_test_official, predizioni_test, zero_division=0))
