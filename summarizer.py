import re
import numpy as np
from collections import defaultdict

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves"
}

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def split_into_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    cleaned = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) > 0 and len(re.findall(r'\w+', s_clean)) > 0:
            cleaned.append(s_clean)
    return cleaned


def remove_stopwords_and_clean(sentence: str) -> str:
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', sentence.lower())
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(filtered_tokens)


def compute_similarity_matrix(sentences: list[str]) -> np.ndarray:
    n = len(sentences)
    if n <= 1:
        return np.zeros((n, n))

    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=5000
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(similarity_matrix, 0)
            return similarity_matrix
        except Exception:
            pass

    return _pure_python_similarity_matrix(sentences)


def _pure_python_similarity_matrix(sentences: list[str]) -> np.ndarray:
    n = len(sentences)
    docs = [remove_stopwords_and_clean(s).split() for s in sentences]
    
    vocab = sorted(list(set(w for d in docs for w in d)))
    if not vocab:
        return np.zeros((n, n))
    
    doc_freq = defaultdict(int)
    for d in docs:
        for w in set(d):
            doc_freq[w] += 1
            
    vectors = []
    for d in docs:
        tf = defaultdict(int)
        for w in d:
            tf[w] += 1
        vec = []
        for w in vocab:
            tfidf = (tf[w] / max(len(d), 1)) * np.log((n + 1) / (doc_freq[w] + 1) + 1)
            vec.append(tfidf)
        vectors.append(vec)
        
    vecs = np.array(vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vecs = vecs / norms
    
    sim_matrix = normalized_vecs @ normalized_vecs.T
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


def compute_two_document_similarity(doc1: str, doc2: str) -> float:
    if not doc1.strip() or not doc2.strip():
        return 0.0
        
    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(sim)
        except Exception:
            pass

    sim_matrix = _pure_python_similarity_matrix([doc1, doc2])
    return float(sim_matrix[0, 1]) if sim_matrix.shape == (2, 2) else 0.0


def run_pagerank(
    similarity_matrix: np.ndarray, 
    damping_factor: float = 0.85, 
    max_iter: int = 100, 
    tol: float = 1e-6
) -> np.ndarray:
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    scores = np.ones(n) / n

    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    transition = np.divide(
        similarity_matrix,
        row_sums,
        out=np.zeros_like(similarity_matrix),
        where=row_sums != 0
    )

    dangling = (row_sums.ravel() == 0)

    for _ in range(max_iter):
        new_scores = (
            (1.0 - damping_factor) / n
            + damping_factor * (
                transition.T @ scores
                + scores[dangling].sum() / n
            )
        )

        if np.linalg.norm(new_scores - scores, ord=1) < tol:
            scores = new_scores
            break

        scores = new_scores

    return scores


def map_reduce_character_count(text: str) -> dict[str, int]:
    mapped = []
    for ch in text:
        if ch.isalpha():
            mapped.append((ch.lower(), 1))

    reduced = defaultdict(int)
    for char, count in mapped:
        reduced[char] += count

    return dict(sorted(reduced.items()))


def summarize(text: str, num_sentences: int = 3, damping_factor: float = 0.85) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "summary": "",
            "sentences": [],
            "scores": [],
            "ranked_sentences": [],
            "top_indices": [],
            "similarity_matrix": np.array([]),
            "original_word_count": 0,
            "summary_word_count": 0,
            "compression_ratio": 0.0
        }

    sentences = split_into_sentences(text)
    total_sentences = len(sentences)
    original_words = sum(len(s.split()) for s in sentences)

    if total_sentences == 0:
        return {
            "summary": "",
            "sentences": [],
            "scores": [],
            "ranked_sentences": [],
            "top_indices": [],
            "similarity_matrix": np.array([]),
            "original_word_count": 0,
            "summary_word_count": 0,
            "compression_ratio": 0.0
        }

    if total_sentences <= num_sentences:
        summary_text = " ".join(sentences)
        summary_words = sum(len(s.split()) for s in sentences)
        return {
            "summary": summary_text,
            "sentences": sentences,
            "scores": [1.0 / total_sentences] * total_sentences,
            "ranked_sentences": [{"rank": i + 1, "index": i, "score": 1.0 / total_sentences, "text": s} for i, s in enumerate(sentences)],
            "top_indices": list(range(total_sentences)),
            "similarity_matrix": np.zeros((total_sentences, total_sentences)),
            "original_word_count": original_words,
            "summary_word_count": summary_words,
            "compression_ratio": round((1 - summary_words / max(original_words, 1)) * 100, 1)
        }

    similarity_matrix = compute_similarity_matrix(sentences)
    scores = run_pagerank(similarity_matrix, damping_factor=damping_factor)

    ranked_indices = np.argsort(scores)[::-1]
    ranked_sentences = [
        {
            "rank": rank + 1,
            "index": int(idx),
            "score": float(scores[idx]),
            "text": sentences[idx]
        }
        for rank, idx in enumerate(ranked_indices)
    ]

    top_indices = sorted(ranked_indices[:num_sentences])
    summary_sentences = [sentences[i] for i in top_indices]
    summary_text = " ".join(summary_sentences)

    summary_words = sum(len(s.split()) for s in summary_sentences)
    compression = round((1 - (summary_words / max(original_words, 1))) * 100, 1)

    return {
        "summary": summary_text,
        "sentences": sentences,
        "scores": scores.tolist(),
        "ranked_sentences": ranked_sentences,
        "top_indices": top_indices,
        "similarity_matrix": similarity_matrix,
        "original_word_count": original_words,
        "summary_word_count": summary_words,
        "compression_ratio": compression
    }
