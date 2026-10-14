# Extractive Document Summarizer

An extractive text summarization system and interactive web application built with Python and Streamlit. The project applies foundational Information Retrieval (IR) algorithms to analyze, rank, and summarize textual documents based on sentence salience.

---

## 📌 Project Overview

Unlike abstractive summarization (which generates new paraphrased sentences), **extractive summarization** identifies and extracts the most informative, central sentences directly from the original document while preserving their original chronological order.

This project implements an end-to-end Information Retrieval pipeline:
1. **Sentence Boundary Detection & Word Tokenization**
2. **Text Normalization & Stop Word Removal**
3. **Vector Space Modeling via TF-IDF (Term Frequency - Inverse Document Frequency)**
4. **Pairwise Cosine Similarity Computation** to construct a sentence-connectivity graph
5. **Graph-based Sentence Scoring via PageRank (TextRank)**
6. **Chronological Sentence Reordering & Metric Evaluation**
7. **Document-to-Document Semantic Similarity Measurement**

---

## 📚 Topics Covered

| Topic | Description | Implementation Details |
|---|---|---|
| **Text Preprocessing** | Cleaning raw, unstructured text for retrieval tasks | Regex-based sentence boundary segmentation, lowercasing, alphanumeric token extraction, and stop-word filtering |
| **Vector Space Model (VSM)** | Representing text units as numeric vectors in multidimensional space | Vocabulary construction, Term Frequency ($TF$), and Inverse Document Frequency ($IDF$) weighting |
| **Cosine Similarity** | Measuring directional similarity between vectors independent of document length | Dot product of $L_2$-normalized TF-IDF vectors, generating a symmetric pairwise sentence similarity matrix |
| **PageRank (TextRank)** | Unsupervised graph-based ranking algorithm | Sentences modeled as graph nodes; cosine similarities as weighted edges; power iteration with damping factor ($d = 0.85$) |
| **Document Similarity** | Evaluating semantic overlap between two independent documents | TF-IDF vectorization and cosine similarity scoring (0.0 to 1.0) |
| **Evaluation Metrics** | Quantifying summarization effectiveness | Original word count, summary word count, and compression ratio |

---

## 🔄 End-to-End Processing Flow

```
                      +----------------------------------+
                      |         Raw Input Text           |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |   Sentence Splitting & Regex     |
                      |   [S1, S2, S3, ..., Sn]          |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |  Lowercasing & Stop Word Filter  |
                      |  Filtered Tokens per Sentence    |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |     TF-IDF Vectorization         |
                      |  Sentence Vectors in R^|V|       |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |   Cosine Similarity Matrix       |
                      |   W(i, j) = cos(v_i, v_j)        |
                      |   (Diagonal zeroed: W(i, i) = 0) |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |   PageRank Graph Formulation     |
                      |   Row-normalized transition P    |
                      |   Power Iteration: p^(t+1)       |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |   Sentence Scoring & Ranking     |
                      |   Scores: [score_1, ..., score_n]|
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |   Top-K Extraction & Re-ordering |
                      |   Restore Chronological Order    |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |         Final Summary            |
                      |    + Compression Statistics      |
                      +----------------------------------+
```

### Detailed Pipeline Stages

#### 1. Sentence Segmentation & Tokenization
The raw text is split into distinct sentences using boundary punctuation (`.`, `!`, `?`) followed by whitespace or line breaks. Sentences without sufficient alphanumeric content are pruned.

#### 2. Stop Word Removal & Normalization
Common English function words (*the, is, at, which, on, for, ...*) contribute high frequency but carry low semantic discriminative power. These are removed from each sentence using a curated stop-word lexicon, and remaining tokens are converted to lowercase.

#### 3. Vector Space Representation (TF-IDF)
Each sentence $S_i$ is mapped to a vector in a shared $|V|$-dimensional vocabulary space:
- **Term Frequency ($TF$)**: Measures the frequency of word $t$ inside sentence $S_i$:
  $$\text{TF}(t, S_i) = \frac{\text{count}(t, S_i)}{|S_i|}$$
- **Inverse Document Frequency ($IDF$)**: Downweights terms appearing across many sentences:
  $$\text{IDF}(t, D) = \ln\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1$$
  where $N$ is the total sentence count, and $\text{DF}(t)$ is the number of sentences containing term $t$.
- **TF-IDF Weight**:
  $$\text{TF-IDF}(t, S_i) = \text{TF}(t, S_i) \times \text{IDF}(t, D)$$

Each sentence vector $\mathbf{v}_i$ is then normalized using Euclidean ($L_2$) normalization:
$$\hat{\mathbf{v}}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|_2}$$

#### 4. Cosine Similarity Matrix
The semantic similarity between two sentences $S_i$ and $S_j$ is given by the cosine of the angle between their normalized vectors:
$$\text{Sim}(S_i, S_j) = \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\|_2 \|\mathbf{v}_j\|_2}$$

Properties of the similarity matrix $W$:
- **Symmetric**: $W_{ij} = W_{ji}$
- **Values**: $0 \le W_{ij} \le 1$
- **No Self-Loops**: $W_{ii} = 0$. Removing self-loops prevents a sentence from artificially inflating its own centrality score during graph traversal.

#### 5. PageRank (TextRank) Graph Algorithm
The document is modeled as an undirected, weighted graph $G = (V, E)$, where each sentence is a node and each edge weight equals the cosine similarity between the respective pair of sentences.

- The stochastic transition matrix $M$ is obtained by row-normalizing $W$:
  $$M_{ij} = \frac{W_{ij}}{\sum_{k=1}^N W_{ik}}$$
  If a row sums to zero (a disconnected/dangling sentence), probability mass is distributed evenly across all nodes.
- **Power Iteration Update**:
  $$\mathbf{p}^{(t+1)} = \frac{1 - d}{N} \mathbf{1} + d \left( M^T \mathbf{p}^{(t)} + \frac{\sum_{i \in \text{dangling}} p_i^{(t)}}{N} \mathbf{1} \right)$$
  where:
  - $d$ is the damping factor (default $0.85$), representing the probability of continuing a random walk rather than jumping ("teleporting") to a random sentence.
  - $\mathbf{p}^{(t)}$ is the vector of sentence scores at iteration $t$.
  - Iteration continues until the $L_1$ norm $\|\mathbf{p}^{(t+1)} - \mathbf{p}^{(t)}\|_1 < 10^{-6}$ or maximum iterations ($100$) are reached.

#### 6. Summary Assembly & Re-ordering
The top $K$ sentences with the highest PageRank scores are selected. To maintain logical narrative flow and readability, these selected sentences are sorted back into their **original occurrence order** within the text.

---

## 💡 Concrete Example Walkthrough

Consider a 3-sentence input text:

> **Sentence 1 ($S_1$):** *"Artificial intelligence transforms modern healthcare."*  
> **Sentence 2 ($S_2$):** *"Healthcare systems use machine learning and artificial intelligence."*  
> **Sentence 3 ($S_3$):** *"Doctors analyze patient scans with machine learning."*  

### Step 1: Preprocessing & Stop Word Filtering
- $S_1$ cleaned tokens: `['artificial', 'intelligence', 'transforms', 'modern', 'healthcare']`
- $S_2$ cleaned tokens: `['healthcare', 'systems', 'use', 'machine', 'learning', 'artificial', 'intelligence']`
- $S_3$ cleaned tokens: `['doctors', 'analyze', 'patient', 'scans', 'machine', 'learning']`

### Step 2: Vocabulary & Overlap Analysis
Notice the term sharing across sentences:
- **$S_1$ and $S_2$ share:** `"artificial"`, `"intelligence"`, `"healthcare"` (High overlap)
- **$S_2$ and $S_3$ share:** `"machine"`, `"learning"` (Moderate overlap)
- **$S_1$ and $S_3$ share:** None (No shared vocabulary terms)

### Step 3: Cosine Similarity Matrix Computation
After computing normalized TF-IDF vectors, the pairwise cosine dot products are evaluated:

| | $S_1$ | $S_2$ | $S_3$ |
|---|---|---|---|
| **$S_1$** | $0.0000$ | $0.5841$ | $0.0000$ |
| **$S_2$** | $0.5841$ | $0.0000$ | $0.3418$ |
| **$S_3$** | $0.0000$ | $0.3418$ | $0.0000$ |

*(Note: Diagonal elements are explicitly zeroed).*

### Step 4: PageRank Scoring
- $S_2$ has outgoing connections to both $S_1$ and $S_3$.
- $S_1$ connects only to $S_2$.
- $S_3$ connects only to $S_2$.

Because $S_2$ shares information with both other sentences, it functions as the central bridge/hub node in the document graph. The PageRank power iteration converges to:

| Rank | Sentence | PageRank Score | Role in Document |
|:---:|---|:---:|---|
| **1** | **$S_2$** | **0.4286** | **Highest Centrality (Core Summary Sentence)** |
| 2 | $S_1$ | 0.3152 | High similarity to $S_2$ |
| 3 | $S_3$ | 0.2562 | Moderate similarity to $S_2$ |

### Step 5: Final Summary Generation
If $K = 1$ is requested:
> *"Healthcare systems use machine learning and artificial intelligence."*

If $K = 2$ is requested:
Top sentences are $S_2$ and $S_1$. Restoring original chronological order ($S_1 \to S_2$):
> *"Artificial intelligence transforms modern healthcare. Healthcare systems use machine learning and artificial intelligence."*

---

## 🛠️ Project Structure

```
IR-Mini Project/
├── app.py              # Streamlit web application & user interface
├── summarizer.py       # Core algorithms (TF-IDF, PageRank, Cosine Similarity)
├── requirements.txt    # Project dependencies
└── README.md           # Documentation and methodology guide
```

### Component Details
- `summarizer.py`:
  - `split_into_sentences(text)`: Splits text into clean sentences via regex.
  - `remove_stopwords_and_clean(sentence)`: Normalizes tokens and strips stop words.
  - `compute_similarity_matrix(sentences)`: Generates TF-IDF vectors and computes pairwise cosine similarity with zeroed diagonal.
  - `run_pagerank(similarity_matrix, damping_factor)`: Implements the PageRank power iteration algorithm.
  - `summarize(text, num_sentences, damping_factor)`: End-to-end pipeline returning summary text, ranked sentences, word counts, and compression ratio.
  - `compute_two_document_similarity(doc1, doc2)`: Computes TF-IDF cosine similarity between two arbitrary text passages.
- `app.py`:
  - Interactive UI for adjusting summary length ($K$) and damping factor ($d$).
  - Summary display with compression metrics.
  - Inspection tab showing the **Top 5 PageRank scored sentences**.
  - Inspection tab showing the **Top 5 preprocessed sentences** (before and after stop-word removal).
  - Document-to-document similarity comparator.

---

## 🚀 Setup & Execution

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Summary of Features

- **Extractive Precision**: Extracts salient sentences without hallucination or factual drift.
- **Top 5 Visual Inspections**: Dedicated breakdown tabs for top 5 PageRank scores and top 5 stop-word removal previews.
- **Adjustable Parameters**: Live slider control over summary sentence count ($1 \text{--} 5$) and damping factor ($0.50 \text{--} 0.95$).
- **Offline & Fallback Support**: Built-in pure-Python TF-IDF fallback if Scikit-Learn is unavailable.
- **Two-Document Similarity**: Built-in utility to compare any two text blocks using the Vector Space Model.
