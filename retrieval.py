import numpy as np
from datetime import datetime
import pinecone


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two input vectors
    :param a: 1-D array object
    :param b: 1-D array object
    :return: scalar value representing the cosine similarity between the input vectors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def exponential_decay(
    earlier_time: datetime, current_time: datetime, decay_rate=0.01
) -> float:
    """
    Calculates an exponential decay score based on the time difference between two timestamps using log space.
    :param earlier_time: The timestamp indicating when the memory was last accessed.
    :param current_time: The current timestamp.
    :param decay_rate: The rate of decay. Defaults to 0.01.

    :returns: The calculated exponential decay score.
    """
    time_difference = (current_time - earlier_time).total_seconds()
    score = 1 / (1 + decay_rate * time_difference)
    return score


def recency_score(record_recent_access: datetime, cur_time: datetime) -> float:
    """
    Calculates the recency value a record has to the current query.
    Leverages exponential decay from the time the current query was posed to assign a score.
    :param record_recent_access: When the current record was last accessed
    :param cur_time: The time when the user's query was posed to the character
    :return: The recency score of the presented record
    """
    return exponential_decay(record_recent_access, cur_time)


def importance_score(record: dict) -> float:
    """
    Retrieves the importance (poignancy) of the presented record
    :param record: The current record to determine an importance score for
    :return: The poignancy of the presented record
    """
    return record["metadata"]["poignancy"]


def relevance_score(record_embedding: list, query_embedding: list) -> float:
    """
    Calculates the relatedness of the presented record and the user's query
    :param record_embedding: The embedding representation of the text associated with the current record
    :param query_embedding: The embedding representation of the user's query
    :return: The relevance score of the presented record
    """
    return cos_sim(np.array(record_embedding), np.array(query_embedding))


def context_retrieval(
    namespace: str,
    query_embedding: list[float],
    n: int,
    index_name: str = "chatnpc-index",
) -> list[str]:
    """
    Ranks character memories by a retrieval score.
    Retrieval score calculated by multiplying their importance, recency, and relevance scores together.
    Selects the top n of these records to be used as context when replying to the user's query.
    :return: Text from the n memories
    """
    # get index to query for records
    index: pinecone.Index = pinecone.Index(index_name)
    # count number of vectors in the namespace
    total_vectors: int = index.describe_index_stats()["namespaces"][namespace][
        "vector_count"
    ]
    # query Pinecone and get all records in namespace
    responses = index.query(
        query_embedding,
        top_k=total_vectors + 1,  # +1 to get all vectors in a namespace
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "response"}},
            ]
        },
    )  # don't need to return values cuz we get score

    # find current access time
    cur_time = datetime.now()
    # calculate retrieval score and keep track of record IDs
    # score_id_pairs format: [SCORE, RECORD]
    score_id_pairs: list = [
        (
            recency_score(cur_time, record["metadata"]["last_accessed"])
            * importance_score(record)
            * record["score"],
            record,
        )
        for record in responses["matches"]
    ]
    # sort records by retrieval score
    sorted_score_id_pairs = sorted(
        score_id_pairs, key=lambda pair: pair[0], reverse=True
    )

    # for _, record in sorted_score_id_pairs:
    #     print(record['metadata']['text'])

    # select top n memories
    top_records: list[dict] = []
    top_context: list[str] = []
    for score, record in sorted_score_id_pairs[:n]:
        top_records.append(record)
        top_context.append(record["metadata"]["text"])

    # update records with new access times
    for record in top_records:
        index.update(
            id=record["id"],
            set_metadata={"last_accessed": str(cur_time)},
            namespace=namespace,
        )

    return top_context
