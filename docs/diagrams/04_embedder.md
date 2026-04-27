# Embedder State Machine

The embedder (`embedder.py`) prepends task-specific prefixes required by `nomic-embed-text-v1.5`'s asymmetric search design before calling the fastembed `TextEmbedding` model. `embed_documents()` is used at ingest time (prefix: `search_document: `); `embed_query()` is used at query time (prefix: `search_query: `). Both delegate to the `get_embedder()` singleton from `resources.py`. The state machine is trivially linear; the only branching is empty-input guards and error wrapping.

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> EmbedDocumentsPath : embed_documents(texts) called
    Idle --> EmbedQueryPath : embed_query(query) called

    state EmbedDocumentsPath {
        [*] --> CheckDocTexts
        CheckDocTexts --> ReturnEmptyArray : texts list is empty
        CheckDocTexts --> PrependDocPrefix : texts non-empty

        ReturnEmptyArray --> [*] : returns np.empty((0, 768))

        PrependDocPrefix --> CallingEmbedder : "search_document: " + each text

        CallingEmbedder --> VectorCountMismatch : len(vecs) != len(texts)
        CallingEmbedder --> EmbedDocsDone : vecs returned, count matches

        VectorCountMismatch --> [*] : raises EmbeddingError
        EmbedDocsDone --> [*] : returns float32 array shape (N, 768)
    }

    state EmbedQueryPath {
        [*] --> CheckQueryText
        CheckQueryText --> EmptyQueryError : query.strip() == ""
        CheckQueryText --> PrependQueryPrefix : query non-empty

        EmptyQueryError --> [*] : raises EmbeddingError("Query must not be empty")

        PrependQueryPrefix --> CallingQueryEmbedder : "search_query: " + query

        CallingQueryEmbedder --> EmbedQueryDone : vector returned
        CallingQueryEmbedder --> EmbedQueryFailed : Exception from fastembed

        EmbedQueryDone --> [*] : returns float32 array shape (768,)
        EmbedQueryFailed --> [*] : raises EmbeddingError
    }

    EmbedDocumentsPath --> [*]
    EmbedQueryPath --> [*]
```
