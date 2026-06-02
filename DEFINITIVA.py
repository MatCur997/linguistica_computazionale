# 0) Importazione librerie

from datasets import load_dataset
import html
import re
import pandas as pd
import emoji
import stanza
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------
# 1) Download del dataset da Hugging Face e preparazione dei dati
print("1) Download e preparazione dei dati")

dataset = load_dataset("evalitahf/sentiment_analysis")

df_train_raw = pd.DataFrame(dataset['train'])
df_test_raw = pd.DataFrame(dataset['test'])


def pulizia_etichette(df):
    # Sanity check sui valori delle colonne
    df_valid = df[df['opos'].astype(str).str.fullmatch(r'[01]') &
                   df['oneg'].astype(str).str.fullmatch(r'[01]')].copy()

    # Conversione del datatype da "str" a "int" valori "stringa"
    df_valid['opos'] = df_valid['opos'].astype(int)
    df_valid['oneg'] = df_valid['oneg'].astype(int)

    # Rimozione dei tweet neutri e misti
    df_valid = df_valid[df_valid['opos'] != df_valid['oneg']].copy()
    df_valid['label'] = df_valid['opos']

    return df_valid


df_train = pulizia_etichette(df_train_raw)
df_test = pulizia_etichette(df_test_raw)

# --------------------------------------------
# 2) Pulizia testuale iniziale tramite RegEx
print("\n2) Pulizia Testuale")

def pulizia_iniziale(text):
    # Decodifica HTML
    text = html.unescape(text)

    # 2. Conversione emoji in testo descrittivo; rimozione dei due punti generati dalla libreria
    text = emoji.demojize(text, language='it')
    text = re.sub(r':([^:]+):', r' \1 ', text)

    # 3. Rimozione dei tag specifici del dataset
    text = re.sub(r'<URL>', ' ', text)
    text = re.sub(r'<MENTION_\d+>', ' ', text)

    # 4. Conversione del "#" in "htag_" per gli hashtag; rende tutto il blocco minuscolo;
    #    se trova un apostrofo, lo rimuove
    text = re.sub(r"#(\w+(?:'\w+)?)", lambda m: "htag_" + m.group(1).lower().replace("'", ""), text)

    # 5. Rimozione di eventuali cancelletti isolati
    text = re.sub(r'#', ' ', text)

    # 6. Rimozione di tag di retweet
    text = re.sub(r'(?i)\brt\b\s*@?', ' ', text)

    # 7. Rimozione di spazi extra e di spazi a inizio e fine stringa
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df_train['text_clean'] = df_train['text'].apply(pulizia_iniziale)
df_test['text_clean'] = df_test['text'].apply(pulizia_iniziale)

# --------------------------------------------
# 3) Con Stanza, rimozione delle sia delle stopwords che delle stopwords di dominio;
#    Dependency Parsing utile alla propagazione semantica delle parole di negazione
print("\n3) Estrazione sintattica con Stanza")

nltk.download('stopwords', quiet=True)
stop_words_it = set(stopwords.words('italian'))

# 1. Definizione delle parole di negazione
parole_negazione = {"non", "mai", "nessuno", "senza", "neanche", "nemmeno", "neppure", "né", "ne'"}

parole_avversative = {"ma", "però"}

# 2. Definizione delle stopwords di dominio (Topic Bias)
stop_words_domain = {
    # 1. Blocco politico-istituzionale
    "grillo", "htag_grillo", "beppe", "htag_beppe", "comico", "htag_gri", "htag_grillini",
    "monti", "htag_monti", "htag_oramonti", "htag_supermario", "htag_mario",
    "berlusconi", "htag_berlusconi", "htag_doposilvio", "htag_308",
    "pd", "htag_pd",
    "pdl", "htag_pdl", "htag_abc", "alfano",
    "lega", "htag_lega", "bossi", "htag_bossi",
    "passera", "htag_passera", "sapia", "htag_sapia",
    "pisapia", "htag_pisapia",
    "idv", "htag_idv", "htag_alfano", "dipietro", "htag_dipietro",
    "napolitano", "htag_napolitano", "dalema", "htag_dalema",
    "bersani", "htag_bersani",
    "calder", "htag_calder", "htag_severino", "htag_alemanno", "htag_mussi",
    "governo", "htag_governo", "htag_gover", "htag_parlamento", "htag_colle",
    "htag_sinistra", "htag_partiti", "htag_terzopolo", "htag_politici",
    "htag_democrazia", "htag_liberalizzazioni", "htag_europa", "htag_sottosegretari",
    "htag_conferenzastampa", "htag_ministro", "htag_casini",

    # 2. Blocco economico-sociale
    "manovra", "htag_manovra", "tassa", "banca", "massone",
    "scuola", "htag_labuonascuola", "htag_pensioni", "htag_equità",
    "htag_postofisso", "htag_monotonia", "htag_ilg", "htag_populismo",
    "htag_erroredrammatico", "htag_imu", "htag_economia", "htag_politica",
    "htag_privatizzazioni", "htag_rifiuti", "htag_spread", "htag_patrimoniale",

    # 3. Blocco mediatico
    "htag_serviziopubblico", "htag_piazzapulita", "htag_agorarai",
    "htag_ballarò", "htag_chetempochefa", "adnkronos", "htag_ottoemezzo",
    "htag_santoro", "htag_matrix", "htag_ultimamar", "htag_news", "htag_notizie",
    "htag_fb", "htag_twitter", "htag_robinson", "htag_portaaporta", "htag_mediaset",
    "htag_ballaró", "htag_ballaro",

    # 4. Blocco miscellaneo
    "novembre", "niall", "htag_forzamilan", "htag_desenzanodelgarda",
    "htag_itali", "htag_giorgio", "htag_rt", "htag_ff", "htag_ditelavostra",
    "htag_cotechinoelenticchie", "htag_ama", "htag_vogliodiregraziea",
    "htag_la", "htag_fatepresto", "htag_m5stour", "htag_laresadeiconti",
    "mediaset", "casini", "ministro",  "vaticano", "htag_vaticano",
    "spread",  "patrimoniale", "crisi", "htag_crisi"
}

# 3. Insieme totale delle stopwords
stop_words_it = (stop_words_it - parole_negazione - parole_avversative) | stop_words_domain

# Definizione della Pipeline
nlp = stanza.Pipeline('it', processors='tokenize,pos,lemma,depparse', use_gpu=True)

def estrazione_features_doc(doc):
    final_tokens = []
    for sentence in doc.sentences:
        id_negati = set()

        # Trova le negazioni
        for word in sentence.words:
            if word.text.lower() in parole_negazione:
                head_id = word.head
                id_negati.add(head_id)
                for child in sentence.words:
                    if child.head == head_id and child.deprel in ["obj", "xcomp", "conj"] and child.text.lower() not in parole_negazione:
                        id_negati.add(child.id)

        # Estrazione
        for word in sentence.words:
            testo_token = word.text.lower()

            # Controllo se il token si trova fra le stopwords
            if testo_token in stop_words_it:
                continue

            # 2. Controllo e conservazione degli hashtag (tramite "htag_" iniziale)
            if testo_token.startswith("htag_"):
                final_tokens.append(testo_token)
                continue

            # 3. Rimozione delle stopwords di negazione
            if testo_token in parole_negazione:
                continue

            # Rimozionde della punteggiatura, eccetto "!" e "?"
            if word.upos == "PUNCT":
                if testo_token in ["!", "?"]:
                    final_tokens.append(testo_token)
                continue

            # Lemmatizzazione
            lemma = word.lemma if word.lemma else testo_token

            # Aggiunta del suffisso "_NEG"
            if word.id in id_negati:
                lemma += "_NEG"

            final_tokens.append(lemma)

    return " ".join(final_tokens)

# Definizione di due batch, per ottimizzare i tempi
print("Processamento del batch del train set")
in_docs_train = [stanza.Document([], text=t) for t in df_train['text_clean'].tolist()]
out_docs_train = nlp(in_docs_train)
df_train['text_ready'] = [estrazione_features_doc(doc) for doc in out_docs_train]

print("Processamento del batch del test set")
in_docs_test = [stanza.Document([], text=t) for t in df_test['text_clean'].tolist()]
out_docs_test = nlp(in_docs_test)
df_test['text_ready'] = [estrazione_features_doc(doc) for doc in out_docs_test]

# --------------------------------------------
# 4) Ispezione a campione dei risultati del pre-processing
print("\n4) Ispezione visiva a campione dei risultati del pre-processing")

# Estrazione di 30 tweet dal train set
campione = df_train.sample(n=30, random_state=42)

for index, row in campione.iterrows():
    print(f"Tweet originale: {row['text']}")
    print(f"Tweet dopo la pulizia: {row['text_clean']}")
    print(f"Tweet dopo la pipeline Stanza: {row['text_ready']}")
    print(f"Classe associata al Tweet: {'Positiva (1)' if row['label'] == 1 else 'Negativa (0)'}")
    print("-" * 70)


# --------------------------------------------
# 5) Fase di Machine Learning: combinazione BoW e Tf-Idf, Naive Bayes e Logistic Regression
print("\n5) Machine Learning e confronto sul f1-score")

# Separazione "text" e "target"
X_train_full = df_train['text_ready'].fillna('')
y_train_full = df_train['label'].astype(int)
X_test_final = df_test['text_ready'].fillna('')
y_test_final = df_test['label'].astype(int)

# Definizione della Pipeline di base
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

# Definizione dello spazio di ricerca per le 4 COMBINAZIONI (Il Quadrato Magico)
param_grid = [

    # Combinazione 1: Bag of Words + Naive Bayes
    {
        'vect': [CountVectorizer(token_pattern=r'\S+')],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_features': [3000, 5000, None],
        'vect__min_df': [2, 3],
        'vect__max_df': [0.4, 0.6, 1.0],

        'clf': [MultinomialNB()],
        'clf__alpha': [0.1, 1.0, 2.0, 5.0],
        'clf__fit_prior': [True, False]
    },

    # Combinazione 2: TF-IDF + Naive Bayes
    {
        'vect': [TfidfVectorizer(token_pattern=r'\S+')],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_features': [3000, 5000, None],
        'vect__min_df': [2, 3],
        'vect__max_df': [0.4, 0.6, 1.0],

        'clf': [MultinomialNB()],
        'clf__alpha': [0.1, 1.0, 2.0, 5.0],
        'clf__fit_prior': [True, False]
    },

    # Combinazione 3: Bag of Words + Logistic Regression
    {
        'vect': [CountVectorizer(token_pattern=r'\S+')],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_features': [3000, 5000, None],
        'vect__min_df': [2, 5],

        'clf': [LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.5, 1.0, 5.0],
        'clf__class_weight': ['balanced', None]
    },

    # Combinazione 4: TF-IDF + Logistic Regression
    {
        'vect': [TfidfVectorizer(token_pattern=r'\S+')],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_features': [3000, 5000, None],
        'vect__min_df': [2, 5],

        'clf': [LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.5, 1.0, 5.0],
        'clf__class_weight': ['balanced', None]
    }
]

print("Grid Search per trovare i migliori parametri dei 4 modelli, su una 5-Fold Cross-Validation")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train_full, y_train_full)

# Conservazione del modello migliore, coi migliori parametri
best_overall_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Estrazione dei nomi del vettorializzatore migliore e del classificatore migliore
vect_name = "TF-IDF" if isinstance(best_overall_model.named_steps['vect'], TfidfVectorizer) else "Bag of Words (CountVectorizer)"
clf_name = "Logistic Regression" if isinstance(best_overall_model.named_steps['clf'], LogisticRegression) else "Naive Bayes"

print("\nRisultato della comparazione dei modelli:")
print(f"Vettorializzatore migliore: {vect_name}")
print(f"Classificatore migliore: {clf_name}")
print(f"F1-Macro: {grid_search.best_score_:.4f}")
print(f"Parametri migliori: {best_params}")

# --------------------------------------------
# 6) Prova sul test set e scelta della soglia
print("\nProva sul test set col modello migliore")

probability = best_overall_model.predict_proba(X_test_final)
prob_classe_1 = probability[:, 1]

# Siccome il classificatore vincente si è dimostrato essere la Logistic Regression,
# si lavora sulla sua soglia decisionale. Viene applicato un Threshold Moving dal 0.50 di default al 0.70, per rendere
# il modello più conservativo, cercando i diminuire i Falsi Positivi derivanti da frase ironiche.

moved_threshold = 0.70

y_pred_final = (prob_classe_1 >= moved_threshold).astype(int)

print(f"\nRisultati sul Test Set (Soglia {moved_threshold * 100}%):")
print(classification_report(y_test_final, y_pred_final))

print("Matrice di Confusione:")
print(confusion_matrix(y_test_final, y_pred_final))

# --------------------------------------------
# 7) Analisi correlazione Falsi Positivi e Ironia
print("\n7) Analisi correlazione Falsi Positivi e Ironia")

# Creazione del DataFrame aggiungendo la colonna 'iro' da df_test
df_errors = pd.DataFrame({
    'Tweet originale': df_test['text'].values,
    'Tweet dopo la pipeline Stanza': X_test_final.values,
    'Classe target': y_test_final.values,
    'Classe predetta': y_pred_final,  # <--- CORRETTO QUI
    "Presenza d'ironia": df_test['iro'].astype(int).values
})

## Falsi Positivi (target 0, previsto 1)
falsi_positivi = df_errors[(df_errors['Classe target'] == 0) & (df_errors['Classe predetta'] == 1)]

## Veri Negativi (target 0, previsto 0) per confrontare
veri_negativi = df_errors[(df_errors['Classe target'] == 0) & (df_errors['Classe predetta'] == 0)]

# Calcolo statistiche percentuali
totale_fp = len(falsi_positivi)
fp_ironici = falsi_positivi["Presenza d'ironia"].sum()
perc_ironici_fp = (fp_ironici / totale_fp) * 100 if totale_fp > 0 else 0

totale_tn = len(veri_negativi)
tn_ironici = veri_negativi["Presenza d'ironia"].sum()
perc_ironici_tn = (tn_ironici / totale_tn) * 100 if totale_tn > 0 else 0

# Risultati
print(f"Risultato correlazione con ironia:")
print(f"- Su {totale_fp} Falsi Positivi, {fp_ironici} contengono ironia ({perc_ironici_fp:.1f}%).")
print(f"- Su {totale_tn} Veri Negativi, l'ironia è presente solo nel {perc_ironici_tn:.1f}%. (Gruppo di controllo)")

# Verifica della correlazione
if perc_ironici_fp > perc_ironici_tn:
    rapporto = perc_ironici_fp / perc_ironici_tn if perc_ironici_tn > 0 else 0
    print(f"L'ironia è {rapporto:.1f} volte più frequente nei tweet dove il modello sbaglia.")

# Creazione di un file contenente i risultati per ispezione
nome_file = "falsi_positivi_con_ironia.csv"
# Ordiniamo il file mettendo prima i tweet ironici, così balzano subito all'occhio; usato il separatore ";" per Excel
falsi_positivi_ordinati = falsi_positivi.sort_values(by="Presenza d'ironia", ascending=False)
falsi_positivi_ordinati.to_csv(nome_file, index=False, sep=';', encoding='utf-8-sig')

print(f"\nFile salvato come {nome_file}")

# --------------------------------------------
# 8) Determinazione delle feature più influenti per classe
print("\n8) Determinazione delle feature più influenti per classe e dei loro pesi")

# Estrazione del vettorializzatore e del classificatore del modello migliore
vect = best_overall_model.named_steps['vect']
clf = best_overall_model.named_steps['clf']
nomi_features = vect.get_feature_names_out()

# Estrazione dei pesi dal modello vincente
pesi = clf.coef_[0]

# Creazione del DataFrame con tutte le parole e i rispettivi pesi
df_features = pd.DataFrame({'Parola': nomi_features, 'Peso': pesi})

# Le 20 parole più influenti per classe
top_classe_1 = df_features.sort_values(by='Peso', ascending=False).head(20)
top_classe_0 = df_features.sort_values(by='Peso', ascending=True).head(20)

print("\n Le 20 parole più influenti per la classe 1:")
for _, row in top_classe_1.iterrows():
    print(f"   {row['Parola']:<15} (Peso: {row['Peso']:+.4f})")

print("\n Le 20 parole più influenti per la classe 0:")
for _, row in top_classe_0.iterrows():
    print(f"   {row['Parola']:<15} (Peso: {row['Peso']:+.4f})")

# Analisi degli hashtag
df_hashtags = df_features[df_features['Parola'].str.startswith('htag_')]
top_htag_1 = df_hashtags.sort_values(by='Peso', ascending=False).head(20)
top_htag_0 = df_hashtags.sort_values(by='Peso', ascending=True).head(20)

print("\n I 20 hashtag più influenti per la classe 1:")
for _, row in top_htag_1.iterrows():
    print(f"   {row['Parola']:<20} (Peso: {row['Peso']:+.4f})")

print("\n I 20 hashtag più influenti per la classe 0:")
for _, row in top_htag_0.iterrows():
    print(f"   {row['Parola']:<20} (Peso: {row['Peso']:+.4f})")

# --------------------------------------------
# 9) Analisi del potere discriminativo di "!" e "?"
print("\n Analisi del potere discriminativo di '!' e '?'")

# Peso dato dal modello
print("\n Pesi assegnati dal modello:")
df_punct = df_features[df_features['Parola'].isin(['!', '?'])]

if df_punct.empty:
    print("I simboli '!' e '?' non sono presenti nel vocabolario del modello.")
else:
    for _, row in df_punct.iterrows():
        segno = "Verso la classe 1" if row['Peso'] > 0 else "Verso la classe 0"
        print(f"   Segno '{row['Parola']}': {row['Peso']:+.4f} -> Spinge {segno}")

# Distribuzione nel training set
print("\n Distribuzione nel training set:")

# Funzione per contare la presenza del token
def contiene_token(testo, token):
    parole = str(testo).split()
    return 1 if token in parole else 0

# Calcolo per '!'
df_train['punto_esclamativo'] = df_train['text_ready'].apply(lambda x: contiene_token(x, '!'))
# Calcolo per '?'
df_train['punto_interrogativo'] = df_train['text_ready'].apply(lambda x: contiene_token(x, '?'))

# Raggruppamento per classe
tot_classe_0 = len(df_train[df_train['label'] == 0])
tot_classe_1 = len(df_train[df_train['label'] == 1])

esc_0 = df_train[df_train['label'] == 0]['punto_esclamativo'].sum()
esc_1 = df_train[df_train['label'] == 1]['punto_esclamativo'].sum()

int_0 = df_train[df_train['label'] == 0]['punto_interrogativo'].sum()
int_1 = df_train[df_train['label'] == 1]['punto_interrogativo'].sum()

print(f"   Punto Esclamativo '!':")
print(f"     - Presente nel {(esc_0/tot_classe_0)*100:.1f}% dei tweet negativi")
print(f"     - Presente nel {(esc_1/tot_classe_1)*100:.1f}% dei tweet positivi")

print(f"   Punto Interrogativo '?':")
print(f"     - Presente nel {(int_0/tot_classe_0)*100:.1f}% dei tweet negativi")
print(f"     - Presente nel {(int_1/tot_classe_1)*100:.1f}% dei tweet positivi")

# Risultato
print("\nRisultato:")
for token, freq_0, freq_1 in [('!', esc_0/tot_classe_0, esc_1/tot_classe_1), ('?', int_0/tot_classe_0, int_1/tot_classe_1)]:
    diff = abs(freq_0 - freq_1)
    if diff > 0.10:
        print(f"   -> il '{token}' ha un potere discriminativo.")
    else:
        print(f"   -> il '{token}' è distribuito uniformemente.")
