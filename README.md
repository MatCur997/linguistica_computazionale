# Sentiment Analysis in Italiano con Naive Bayes e Regressione Logistica

## Descrizione

Questo progetto è stato sviluppato come tesina per l'esame di **Linguistica Computazionale - Modulo A**, tenuto dalla Prof.ssa Combei.

L'obiettivo del lavoro è confrontare le prestazioni di diversi modelli di classificazione supervisionata applicati a un task di **Sentiment Analysis binaria** in lingua italiana.

In particolare, vengono confrontate due tecniche di rappresentazione testuale:

- Bag of Words (BoW)
- TF-IDF (Term Frequency – Inverse Document Frequency)

e due algoritmi di classificazione:

- Naive Bayes
- Regressione Logistica

Le combinazioni ottenute vengono valutate mediante metriche standard di classificazione per determinare quale approccio risulti più efficace nell'identificazione automatica del sentiment.

---

## Obiettivi

Gli obiettivi principali del progetto sono:

- comprendere il funzionamento delle rappresentazioni vettoriali classiche del testo;
- confrontare modelli probabilistici e discriminativi;
- valutare l'impatto della scelta della rappresentazione testuale sulle prestazioni;
- applicare metodologie di valutazione estrinseca a un task NLP reale.

---

## Dataset

Il progetto utilizza due dataset italiani per l'analisi del sentiment:

### SENTIPOLC

SENTIPOLC (SENTIment POLarity Classification) costituisce il dataset principale dello studio.

Tutti i modelli vengono addestrati e valutati utilizzando questo corpus, che contiene testi annotati per il sentiment in lingua italiana. Le prestazioni riportate nelle metriche principali derivano da questo dataset.

### Feel-It

Feel-It viene utilizzato esclusivamente in una fase successiva di valutazione **Cross-Dataset**.

Dopo l'addestramento su SENTIPOLC, i modelli vengono testati su Feel-It senza ulteriore addestramento, con l'obiettivo di valutare la loro capacità di generalizzare a dati provenienti da un corpus simile per tipologia testuale, ma con diversità diacronica e argomentale.

Questo approccio consente di osservare quanto i modelli siano robusti rispetto a variazioni nel dominio dei dati e di verificare se le prestazioni ottenute sul dataset di addestramento si mantengano anche in un contesto esterno.

Entrambi i dataset vengono scaricati automaticamente tramite la libreria `datasets` di Hugging Face.

---

## Metodologia

### Rappresentazione del testo

Sono state considerate due tecniche di vettorializzazione:

#### Bag of Words (BoW)

Rappresenta ciascun documento come un vettore contenente le frequenze dei termini presenti nel corpus: agile dal punto vista dell'implementazione e del costo computazionale, ma ignora contesto e ordine delle parole. Inoltre, dà lo stesso peso a ogni termine, sia ai molto frequenti (in senso trans-documentale), sia a quelli più caratterizzanti e informativi (in senso entropico).

#### TF-IDF

Pesa ogni termine in base alla sua frequenza nel documento e alla sua rarità nel corpus, riducendo l'influenza delle parole particolarmente frequenti nel corpus e, al contempo, assegnando maggiore rilevanza ai termini più caratterizzanti; evidenzia le caratteristiche lessicali più utili alla classificazione.

---

### Modelli di classificazione

#### Naive Bayes

Classificatore probabilistico basato sul teorema di Bayes e sull'assunzione di indipendenza condizionata tra le feature; è semplice e veloce, efficace su dati testuali ed economico dal punto di vista computazionale

#### Regressione Logistica

Modello discriminativo che stima direttamente la probabilità di appartenenza a una classe. Fra i vantaggi, gestisce bene l'alta dimensionalità, come quelli derivanti da rappresentazioni Bag of Words e Tf-Idf. Per di più, i pesi appresi dal modello possono essere analizzati per interpretare l'influenza dei singoli termini sulla classificazione.

---

## Configurazioni sperimentali

Sono state confrontate le seguenti combinazioni:

| Vettorializzazione | Modello |
|-------------------|----------|
| Bag of Words | Naive Bayes |
| TF-IDF | Naive Bayes |
| Bag of Words | Regressione Logistica |
| TF-IDF | Regressione Logistica |

---

## Valutazione

Le prestazioni vengono misurate attraverso:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

La valutazione è di tipo **estrinseco**, poiché i modelli vengono giudicati sulla base delle prestazioni ottenute nel task finale di Sentiment Analysis.

---

## Risultati preliminari

I risultati sono ancora in fase di sviluppo.

Le prime sperimentazioni suggeriscono che la combinazione:

**TF-IDF + Regressione Logistica**

ottiene le prestazioni migliori tra quelle valutate finora.

I risultati definitivi verranno riportati al termine della fase sperimentale.

---

## Tecnologie utilizzate

- Python
- Pandas
- Hugging Face Datasets
- Scikit-learn

---

## Stato del progetto

🚧 Work in Progress

Attualmente il progetto è in fase di sviluppo e validazione sperimentale.

Sono previsti ulteriori test e possibili estensioni, tra cui:

- introduzione di tecniche di preprocessing;
- analisi degli errori

---

## Autore

**Matteo Curiale**

Tesina per il corso di Linguistica Computazionale - Modulo A.

---

## Licenza

Questo progetto è distribuito esclusivamente per finalità didattiche e accademiche.
