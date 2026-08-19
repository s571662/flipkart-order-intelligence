from part3_support_agent.rag import retrieve_policy


# Document-level relevance answer key.
# Each query maps to the policy document(s) that should be considered relevant.
EVAL_QUERIES = [
    {
        "query": "What is the return policy for electronics?",
        "relevant_documents": {
            "02_electronics_returns",
        },
    },
    {
        "query": "How will I receive a refund for a COD order?",
        "relevant_documents": {
            "04_cod_refunds",
        },
    },
    {
        "query": "How long does standard delivery usually take?",
        "relevant_documents": {
            "13_delivery_sla",
        },
    },
    {
        "query": "When is reverse pickup available for my return?",
        "relevant_documents": {
            "14_reverse_pickup_eligibility",
            "08_return_pickup_packaging",
        },
    },
    {
        "query": "What happens if my product arrives damaged or defective?",
        "relevant_documents": {
            "06_damaged_defective_products",
        },
    },
    {
        "query": "Can I return apparel or footwear?",
        "relevant_documents": {
            "01_apparel_footwear_returns",
        },
    },
]


def retrieve_top_documents(query: str, k: int = 3):
    """
    Retrieve chunks, map them back to parent documents,
    and deduplicate so evaluation is performed at document level.
    """
    chunk_results = retrieve_policy(
        query,
        top_k=12,
    )

    documents = []
    seen_document_ids = set()

    for result in chunk_results:
        document_id = result["document_id"]

        if document_id in seen_document_ids:
            continue

        seen_document_ids.add(document_id)

        documents.append({
            "document_id": document_id,
            "score": result["score"],
            "text": result["text"],
        })

        if len(documents) == k:
            break

    return documents


def evaluate_retrieval():
    precision_scores = []
    recall_scores = []

    print("=" * 72)
    print("DOCUMENT-LEVEL RETRIEVAL EVALUATION")
    print("=" * 72)

    for query_number, item in enumerate(EVAL_QUERIES, start=1):
        query = item["query"]
        relevant = item["relevant_documents"]

        retrieved_results = retrieve_top_documents(
            query,
            k=3,
        )

        retrieved = {
            result["document_id"]
            for result in retrieved_results
        }

        hits = len(
            retrieved.intersection(relevant)
        )

        precision_at_3 = hits / 3
        recall_at_3 = hits / len(relevant)

        precision_scores.append(precision_at_3)
        recall_scores.append(recall_at_3)

        print(f"\nQuery {query_number}: {query}")

        print(
            "Relevant documents:",
            sorted(relevant),
        )

        print("Retrieved top-3 documents:")

        for rank, result in enumerate(
            retrieved_results,
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{result['document_id']} "
                f"(score={result['score']:.4f})"
            )

        print(
            f"Hits = {hits}"
        )

        print(
            f"Precision@3 = {hits} / 3 "
            f"= {precision_at_3:.4f}"
        )

        print(
            f"Recall@3 = {hits} / {len(relevant)} "
            f"= {recall_at_3:.4f}"
        )

    average_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    average_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    print("\n" + "=" * 72)
    print("AVERAGE RETRIEVAL METRICS")
    print("=" * 72)

    print(
        "Average Precision@3 = "
        f"{sum(precision_scores):.4f} "
        f"/ {len(precision_scores)} "
        f"= {average_precision:.4f}"
    )

    print(
        "Average Recall@3 = "
        f"{sum(recall_scores):.4f} "
        f"/ {len(recall_scores)} "
        f"= {average_recall:.4f}"
    )

    return {
        "average_precision_at_3": average_precision,
        "average_recall_at_3": average_recall,
    }


if __name__ == "__main__":
    evaluate_retrieval()