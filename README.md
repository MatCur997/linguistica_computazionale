# 🐦 Sentiment Analysis su Twitter

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Stanza](https://img.shields.io/badge/Stanza-Stanford_NLP-red.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-yellow.svg)

**Progetto accademico per l'esame di Linguistica Computazionale**
*Neutralizzazione del Topic Bias e ruolo della sintassi e dell'interpunzione nella Sentiment Analysis su Twitter (EVALITA SENTIPOLC 2016).*

---

## 📖 Descrizione del Progetto

Questo repository contiene una pipeline ibrida di Elaborazione del Linguaggio Naturale (NLP) e Machine Learning per la classificazione binaria del sentiment sui Tweet.

I vettorializzatori tradizionali basati puramente sulla frequenza dei termini (*Bag-of-Words*, *Tf-Idf*) faticano a gestire la sintassi tipica del micro-blogging (brevità, ironia, rumore tematico). Questo progetto cerca di limitare questi fattori tramite:
1. **Dependency Parsing** per la gestione rigorosa della negazione sintattica.
2. **Lexical Normalization** avanzata tramite RegEx.
3. **Mitigazione del Topic Bias** (*Shortcut Learning*) tramite l'eliminazione empirica di entità politiche e mediatiche.

## ✨ Caratteristiche Principali

* **Dataset:** EVALITA 2016 SENTIPOLC, scaricato nativamente tramite la libreria Hugging Face `datasets`.
* **Pre-processing Avanzato:** Traduzione semantica delle emoji, pulizia degli artefatti (`<URL>`, `<MENTION_d+>`), e normalizzazione degli hashtag con elisioni.
* **Motore Linguistico (Stanza):** Analisi sintattica per propagare il suffisso `_NEG` al *verbo padre* e ai suoi costituenti retti in caso di negazione.
* **Machine Learning Ottimizzato:** Classificatore di **Regressione Logistica** accoppiato a **TF-IDF**, ottimizzato tramite *Grid Search* (5-Fold Cross-Validation).
* **Threshold Moving:** Calibrazione della soglia decisionale (dal 0.50 al 0.70) per abbattere i Falsi Positivi derivanti dall'ironia.
* **Error Analysis sull'Ironia:** Script dedicato per misurare la correlazione tra gli errori del modello e i marcatori di ironia presenti nel dataset.

## 📊 Risultati Ottenuti

Il modello vincente (TF-IDF + Logistic Regression con regolarizzazione L2, C=5.0) ha raggiunto sul Test Set i seguenti risultati:

* **Accuracy:** 72%
* **F1-Macro:** 0.7188
* L'analisi di estrazione delle feature ha confermato l'importanza della semantica (es. parola *"problema"* come predittore negativo), del lessico diretto e della pragmatica (il simbolo `?` ha dimostrato potere discriminativo).

## 🚀 Installazione e Setup

### Prerequisiti
Ambiente Python configurato. Le dipendenze principali sono:

```
pip install pandas scikit-learn stanza emoji datasets nltk
```

## 📁 Struttura del Repository

* `main.py`: Script principale contenente la pipeline di pre-processing e il modello di classificazione finale ottimizzato.
* `falsi_positivi_con_ironia.csv`: Output generato automaticamente dallo script per l'ispezione visiva dei tweet sarcastici (Error Analysis).

## 👤 Autore
**Matteo Curiale**
