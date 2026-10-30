"""Experiment: see how embeddings measure meaning, before building the real pipeline."""
import numpy as np
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="tenk-risk-analyst-emira", location="global")
MODEL = "gemini-embedding-2"

docs = [
    "The Company depends on component suppliers located primarily in Asia.",
    "New U.S. export controls restrict sales of advanced chips to China.",
    "Rising interest rates may reduce consumer demand for auto financing.",
    "Our stores are open seven days a week in most locations.",
]
query = "Which companies face supply chain risk?"


def embed(texts):
    """Embed texts one at a time (this model on Vertex AI accepts one input per request)."""
    vectors = []
    for t in texts:
        response = client.models.embed_content(
            model=MODEL,
            contents=t,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        vectors.append(response.embeddings[0].values)
    return np.array(vectors)

doc_vectors = embed([f"title: none | text: {d}" for d in docs])
query_vector = embed([f"task: search result | query: {query}"])[0]

print("Shape of document vectors:", doc_vectors.shape)
print("First 5 numbers of doc 1:", np.round(doc_vectors[0][:5], 4))
print()

# The vectors are normalized (length 1), so a dot product equals cosine similarity
scores = doc_vectors @ query_vector
print(f'Query: "{query}"\n')
for score, doc in sorted(zip(scores, docs), reverse=True):
    print(f"{score:.3f}  {doc}")