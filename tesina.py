import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# --- 1. FUNZIONE DI PREPROCESSING (AGGIORNATA) ---
def pulisci_sentipolc(hf_dataset_split):
    """
    Riceve uno split (train o test) di SENTIPOLC, rimuove neutri e misti,
    e restituisce un DataFrame con colonne 'text' e 'label' (binario).
    """
    df = pd.DataFrame(hf_dataset_split)

    # FIX: Convertiamo le colonne in numeri interi.
    # Se erano stringhe ('1', '0') o booleani (True, False), ora diventano sicuramente 1 e 0.
    df['opos'] = pd.to_numeric(df['opos'], errors='coerce').fillna(0).astype(int)
    df['oneg'] = pd.to_numeric(df['oneg'], errors='coerce').fillna(0).astype(int)

    # Filtriamo solo polarità esclusive (usando il trucco del diverso: XOR)
    df_filtered = df[df['opos'] != df['oneg']].copy()

    # Creiamo il target binario (1 = positivo, 0 = negativo)
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


# --- 2. CARICAMENTO E PREPARAZIONE DEI DATI ---
print("Scaricamento di SENTIPOLC in corso...")
dataset = load_dataset("evalitahf/sentiment_analysis")

# Applichiamo la pulizia sia al train che al test set ufficiale
df_train_full = pulisci_sentipolc(dataset['train'])
df_test_official = pulisci_sentipolc(dataset['test'])

# Facciamo un piccolo check di controllo nel terminale
print(f"Righe rimaste nel Train Set dopo la pulizia: {len(df_train_full)}")
print(f"Righe rimaste nel Test Set dopo la pulizia: {len(df_test_official)}")

X_train_full = df_train_full['text']
y_train_full = df_train_full['label']

X_test_official = df_test_official['text']
y_test_official = df_test_official['label']

# ==============================================================================
# FASE 1: VALIDAZIONE INTERNA (Split 80-20 solo sul Train Set)
# ==============================================================================
print("\n" + "=" * 70)
print(" FASE 1: VALIDAZIONE INTERNA (Tuning sui tweet di Train)")
print("=" * 70)

# Dividiamo il train set in 80% addestramento interno e 20% validazione
X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Vettorizzazione (Fittiamo SOLO sull'80% interno)
vec_bow_int = CountVectorizer()
vec_tfidf_int = TfidfVectorizer()

X_train_int_bow = vec_bow_int.fit_transform(X_train_int)
X_val_int_bow = vec_bow_int.transform(X_val_int)

X_train_int_tfidf = vec_tfidf_int.fit_transform(X_train_int)
X_val_int_tfidf = vec_tfidf_int.transform(X_val_int)

# Inizializziamo i modelli per la validazione
nb_int = MultinomialNB()
lr_int = LogisticRegression(max_iter=1000)

# Valutiamo i modelli sul 20% interno
valuta_modello(nb_int, X_train_int_bow, X_val_int_bow, y_train_int, y_val_int, "Naive Bayes", "Bag-of-Words",
               "VALIDAZIONE")
valuta_modello(nb_int, X_train_int_tfidf, X_val_int_tfidf, y_train_int, y_val_int, "Naive Bayes", "TF-IDF",
               "VALIDAZIONE")
valuta_modello(lr_int, X_train_int_bow, X_val_int_bow, y_train_int, y_val_int, "Regressione Logistica", "Bag-of-Words",
               "VALIDAZIONE")
valuta_modello(lr_int, X_train_int_tfidf, X_val_int_tfidf, y_train_int, y_val_int, "Regressione Logistica", "TF-IDF",
               "VALIDAZIONE")

# ==============================================================================
# FASE 2: TEST UFFICIALE (Addestramento 100% Train -> Test su Test Set Ufficiale)
# ==============================================================================
print("\n\n" + "=" * 70)
print(" FASE 2: VALUTAZIONE FINALE SUL TEST SET UFFICIALE DI SENTIPOLC")
print("=" * 70)

# Vettorizzazione finale (Fittiamo sul 100% del Train Set pulito)
vec_bow_final = CountVectorizer()
vec_tfidf_final = TfidfVectorizer()

# Creiamo le matrici definitive
X_train_full_bow = vec_bow_final.fit_transform(X_train_full)
X_test_off_bow = vec_bow_final.transform(X_test_official)

X_train_full_tfidf = vec_tfidf_final.fit_transform(X_train_full)
X_test_off_tfidf = vec_tfidf_final.transform(X_test_official)

# Inizializziamo nuovi modelli definitivi
nb_final = MultinomialNB()
lr_final = LogisticRegression(max_iter=1000)

# Valutiamo i modelli finali sul Test Set ufficiale
valuta_modello(nb_final, X_train_full_bow, X_test_off_bow, y_train_full, y_test_official, "Naive Bayes", "Bag-of-Words",
               "TEST FINALE")
valuta_modello(nb_final, X_train_full_tfidf, X_test_off_tfidf, y_train_full, y_test_official, "Naive Bayes", "TF-IDF",
               "TEST FINALE")
valuta_modello(lr_final, X_train_full_bow, X_test_off_bow, y_train_full, y_test_official, "Regressione Logistica",
               "Bag-of-Words", "TEST FINALE")
valuta_modello(lr_final, X_train_full_tfidf, X_test_off_tfidf, y_train_full, y_test_official, "Regressione Logistica",
               "TF-IDF", "TEST FINALE")

