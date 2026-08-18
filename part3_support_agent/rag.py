from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"


def load_policy_documents():
    documents = []

    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        documents.append({
            "document_id": file_path.stem,
            "text": text,
        })

    return documents


def split_into_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_policy_chunks():
    documents = load_policy_documents()

    chunks = []

    for document in documents:
        sentences = split_into_sentences(document["text"])

        for index, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{document['document_id']}_chunk_{index}",
                "document_id": document["document_id"],
                "text": sentence,
            })

    return chunks


if __name__ == "__main__":
    documents = load_policy_documents()
    chunks = build_policy_chunks()

    print("Policy documents:", len(documents))
    print("Sentence chunks:", len(chunks))

    print("\nFirst 5 chunks:")

    for chunk in chunks[:5]:
        print(chunk)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROUNDING_THRESHOLD = 0.60


def build_embeddings():
    chunks = build_policy_chunks()

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return chunks, embeddings

def build_faiss_index():
    chunks, embeddings = build_embeddings()

    embedding_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    return chunks, embeddings, index

def retrieve_policy(query: str, top_k: int = 3):
    chunks, _, index = build_faiss_index()

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for rank, chunk_index in enumerate(indices[0]):
        if chunk_index == -1:
            continue

        chunk = chunks[chunk_index]

        results.append({
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "text": chunk["text"],
        })

    return results

def grounded_policy_answer(query: str) -> dict:
    results = retrieve_policy(
        query,
        top_k=3,
    )

    if not results:
        return {
            "grounded": False,
            "score": 0.0,
            "threshold": GROUNDING_THRESHOLD,
            "answer": (
                "I do not have sufficient policy evidence "
                "to answer this question."
            ),
            "sources": [],
        }

    top_score = results[0]["score"]

    if top_score < GROUNDING_THRESHOLD:
        return {
            "grounded": False,
            "score": top_score,
            "threshold": GROUNDING_THRESHOLD,
            "answer": (
                "I do not have sufficient policy evidence "
                "to answer this question."
            ),
            "sources": results,
        }

    answer_text = " ".join(
        result["text"]
        for result in results
        if result["score"] >= GROUNDING_THRESHOLD
    )

    return {
        "grounded": True,
        "score": top_score,
        "threshold": GROUNDING_THRESHOLD,
        "answer": answer_text,
        "sources": results,
    }