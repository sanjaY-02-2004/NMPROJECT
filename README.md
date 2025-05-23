#  Financial Named Entity Recognition (NER)

A hybrid NLP pipeline that extracts entities from financial documents using:

-  **SpaCy NER**
-  **Fine-tuned FinBERT**
-  **Regex-based Rule Matching**

---

##  Project Overview

This project focuses on extracting domain-specific entities like organizations, monetary amounts, dates, and financial events from unstructured financial text using both ML-based and rule-based approaches.

---

##  Components

### 1.  SpaCy NER

- Uses `spacy.blank("en")` for training a custom NER model on labeled financial text.
- Annotated entities are converted into `.spacy` format using `DocBin`.
- Custom entity labels like `ORG`, `MONEY`, `DATE`, etc.

### 2.  Fine-tuned FinBERT

- Utilizes the [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone) model pre-trained on financial corpora.
- Fine-tuned as a token classification model using Hugging Face's `Trainer` API.
- Useful for identifying sentiment-laden and contextual financial tokens.

### 3.  Regex-based Rule Matching

- Uses SpaCy's `Matcher` to detect structured entities (e.g., `$5.3M`, `Q1 2024`, `USD 100K`).
- Great for rule-based identification of predictable patterns.
- Integrated as a custom component in the pipeline.

---



