import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: first embedding vector
        vec_b: second embedding vector

    Returns:
        float: cosine similarity score in range [-1, 1]
    
    Formula:
        cosine_similarity = (A · B) / (||A|| × ||B||)
        
        Where:
        - A · B = dot product (sum of element-wise products)
        - ||A|| = magnitude/norm of A (sqrt of sum of squares)
        - ||B|| = magnitude/norm of B
        
    Example (RAG use case):
        - Query embedding: [0.1, 0.2, 0.3]
        - Doc embedding:   [0.15, 0.25, 0.35]
        - Cosine similarity: ~0.998 (very similar)
    """
    if not vec_a or not vec_b:
        return 0.0
    
    # Compute dot product: sum of element-wise products
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    
    # Compute magnitude (norm) of vec_a: sqrt(sum of squares)
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    
    # Compute magnitude (norm) of vec_b: sqrt(sum of squares)
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (norm_a * norm_b)
