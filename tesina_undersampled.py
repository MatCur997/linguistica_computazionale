import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# 1) Pre-processing del dataset
def pulizia_sentipolc(hf_dataset_split):
    """
    Riceve train o test di SENTIPOLC, rimuove neutri e misti.
    Restituisce un DataFrame con colonne 'text' e 'label' (binario, basato sui valori di 'opos').
    """
    df = pd.DataFrame(hf_dataset_split)

    # Conversione in numeri interi dei valori stringa '0' e '1'
    df['opos'] = pd.to_numeric(df['opos'], errors='coerce').fillna(0).astype(int)
    df['oneg'] = pd.to_numeric(df['oneg'], errors='coerce').fillna(0).astype(int)

    # Filtro tramite XOR di polarità esclusive
    df_filtered = df[df['opos'] != df['oneg']].copy()

    # Creazione del target binario (1 = positivo, 0 = negativo)
    df_filtered['label'] = df_filtered['opos']

    return df_filtered[['text', 'label']]


def valuta_modello(modello, X_train_vec, X_test_vec, y_train, y_test, nome_modello, nome_vettorizzatore, fase):
    """Funzione helper per addestrare e stampare i risultati formattati"""
    print(f"\n{'-' * 55}")
    print(f"[{fase}] {nome_modello} con {nome_vettorizzatore}")
    print(f"{'-' * 55}")

    # Addestramento
    modello.fit(X_train_vec, y_train)

    # Predizione
    predizioni = modello.predict(X_test_vec)

    # Report metriche
    print(classification_report(y_test, predizioni, zero_division=0))

# 2.0 bilanciamento undersampling dei negativi
def bilancia_undersampling(df):
    """
    Riceve un DataFrame, individua la classe minoritaria e fa un
    campionamento casuale (undersampling) della classe maggioritaria.
    Restituisce un DataFrame bilanciato e rimescolato.
    """
    # Separiamo le classi
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]

    # Troviamo la dimensione della classe minoritaria (positivi)
    min_size = len(df_pos)

    # Scegliamo un campione casuale di negativi grande quanto i positivi
    df_neg_undersampled = df_neg.sample(n=min_size, random_state=42)

    # Uniamo i due dataframe e rimescoliamo le righe (frac=1)
    df_bilanciato = pd.concat([df_pos, df_neg_undersampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_bilanciato

# 2) Caricamento e preparazione dei dati
print("Scaricamento di SENTIPOLC in corso...")
dataset = load_dataset("evalitahf/sentiment_analysis")

# Applichiamo la pulizia sia al train che al test set ufficiale
df_train_full_sbilanciato = pulizia_sentipolc(dataset['train'])
df_test_official = pulizia_sentipolc(dataset['test'])

df_train_full = bilancia_undersampling(df_train_full_sbilanciato)

# Check di controllo
print(f"Righe Train Set PRIMA del bilanciamento: {len(df_train_full_sbilanciato)}")
print(f"Righe Train Set DOPO il bilanciamento: {len(df_train_full)} (Positivi: {len(df_train_full[df_train_full['label']==1])}, Negativi: {len(df_train_full[df_train_full['label']==0])})")
print(f"Righe rimaste nel Test Set dopo la pulizia: {len(df_test_official)}")

X_train_full = df_train_full['text']
y_train_full = df_train_full['label']

X_test_official = df_test_official['text']
y_test_official = df_test_official['label']


# ==============================================================================

# Prima Fase: Validazione interna (Split 80-20, Train Set)

# ==============================================================================

print("\n" + "=" * 70)
print(" Prima Fase: validazione interna")
print("=" * 70)

# Split del train set in 80% addestramento interno e 20% validazione
X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Vettorizzazione
vec_bow_int = CountVectorizer()
vec_tfidf_int = TfidfVectorizer()

X_train_int_bow = vec_bow_int.fit_transform(X_train_int)
X_val_int_bow = vec_bow_int.transform(X_val_int)

X_train_int_tfidf = vec_tfidf_int.fit_transform(X_train_int)
X_val_int_tfidf = vec_tfidf_int.transform(X_val_int)

# Inizializzazione dei modelli per la validazione
nb_int = MultinomialNB()
lr_int = LogisticRegression(max_iter=1000)

# Valutazione dei modelli sul 20% interno
valuta_modello(nb_int, X_train_int_bow, X_val_int_bow, y_train_int, y_val_int, "Naive Bayes", "Bag-of-Words",
               "VALIDAZIONE")
valuta_modello(nb_int, X_train_int_tfidf, X_val_int_tfidf, y_train_int, y_val_int, "Naive Bayes", "TF-IDF",
               "VALIDAZIONE")
valuta_modello(lr_int, X_train_int_bow, X_val_int_bow, y_train_int, y_val_int, "Regressione Logistica", "Bag-of-Words",
               "VALIDAZIONE")
valuta_modello(lr_int, X_train_int_tfidf, X_val_int_tfidf, y_train_int, y_val_int, "Regressione Logistica", "TF-IDF",
               "VALIDAZIONE")

# ==============================================================================

# Seconda Fase: valutazione sul Test set ufficiale

# ==============================================================================

print("\n\n" + "=" * 70)
print(" Seconda Fase: valutazione finale sul Test set ufficiale")
print("=" * 70)

# Vettorizzazione finale
vec_bow_final = CountVectorizer()
vec_tfidf_final = TfidfVectorizer()

# Creazione delle matrici
X_train_full_bow = vec_bow_final.fit_transform(X_train_full)
X_test_off_bow = vec_bow_final.transform(X_test_official)

X_train_full_tfidf = vec_tfidf_final.fit_transform(X_train_full)
X_test_off_tfidf = vec_tfidf_final.transform(X_test_official)

# Inizializzazione dei nuovi modelli definitivi
nb_final = MultinomialNB()
lr_final = LogisticRegression(max_iter=1000)

# Valutazionde dei modelli finali sul Test Set ufficiale
valuta_modello(nb_final, X_train_full_bow, X_test_off_bow, y_train_full, y_test_official, "Naive Bayes", "Bag-of-Words",
               "TEST FINALE")
valuta_modello(nb_final, X_train_full_tfidf, X_test_off_tfidf, y_train_full, y_test_official, "Naive Bayes", "TF-IDF",
               "TEST FINALE")
valuta_modello(lr_final, X_train_full_bow, X_test_off_bow, y_train_full, y_test_official, "Regressione Logistica",
               "Bag-of-Words", "TEST FINALE")
valuta_modello(lr_final, X_train_full_tfidf, X_test_off_tfidf, y_train_full, y_test_official, "Regressione Logistica",
               "TF-IDF", "TEST FINALE")

