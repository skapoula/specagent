# Chunker State Machine

The chunker (`chunker.py`) splits Markdown text into token-bounded chunks using a recursive separator hierarchy (`\n\n`, `\n`, ` `, `""`). It uses a module-level tokenizer singleton (double-checked locking via `threading.Lock`) loaded from the local HuggingFace cache. `chunk_with_metadata()` wraps `chunk()` to attach the nearest Markdown heading to each chunk. The state machine is largely linear; the only branching is the tokenizer singleton initialisation and the post-filter fallback for very short documents.

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> CheckingInput : chunk_with_metadata(text) or chunk(text) called

    CheckingInput --> EmptyReturn : text.strip() == ""
    CheckingInput --> LoadingTokenizer : text has content

    EmptyReturn --> [*] : returns []

    state LoadingTokenizer {
        [*] --> CheckTokenizerSingleton
        CheckTokenizerSingleton --> ReturnCachedTokenizer : _tokenizer is not None
        CheckTokenizerSingleton --> AcquireTokenizerLock : _tokenizer is None
        AcquireTokenizerLock --> DoubleCheckTokenizer
        DoubleCheckTokenizer --> ReturnCachedTokenizer : set by another thread
        DoubleCheckTokenizer --> LoadingFromCache : still None inside lock
        LoadingFromCache --> TokenizerReady : AutoTokenizer.from_pretrained() succeeds\n(local_files_only=True)
        LoadingFromCache --> TokenizerMissing : model not in local cache
        TokenizerMissing --> [*] : raises RuntimeError\n("Run specagent download-model")
        TokenizerReady --> ReturnCachedTokenizer
        ReturnCachedTokenizer --> [*]
    }

    LoadingTokenizer --> RecursiveSplitting : tokenizer ready

    state RecursiveSplitting {
        [*] --> TrySeparator
        TrySeparator --> SplitOnDoubleNewline : sep = "\n\n"
        SplitOnDoubleNewline --> MergeSplits : all splits fit in chunk_size_tokens
        SplitOnDoubleNewline --> RecurseDeeper : oversized split found → recurse with "\n"

        RecurseDeeper --> TrySeparator : next separator in hierarchy
        TrySeparator --> SplitOnNewline : sep = "\n"
        TrySeparator --> SplitOnSpace : sep = " "
        TrySeparator --> TokenWindowFallback : sep = "" (character-level last resort)

        TokenWindowFallback --> [*] : encode once, decode sliding windows of chunk_size tokens

        MergeSplits --> [*] : chunks assembled with overlap
    }

    RecursiveSplitting --> FilteringChunks : raw_chunks list produced

    FilteringChunks --> AllChunksBelowMin : all chunks < chunk_min_tokens
    FilteringChunks --> ChunksFiltered : at least one chunk >= chunk_min_tokens

    AllChunksBelowMin --> PreserveRawChunks : fallback: use raw_chunks as-is\n(document shorter than min-token floor)
    PreserveRawChunks --> AttachingHeaders

    ChunksFiltered --> AttachingHeaders

    state AttachingHeaders {
        [*] --> ScanningChunk
        ScanningChunk --> HeadingFound : _HEADER_RE matches in chunk text
        ScanningChunk --> NoHeading : no heading match
        HeadingFound --> UpdateLastHeader : last_header = last matched heading
        UpdateLastHeader --> AppendTuple : (chunk_text, last_header)
        NoHeading --> AppendTuple : (chunk_text, last_header) — inherits previous
        AppendTuple --> ScanningChunk : more chunks
        AppendTuple --> [*] : all chunks processed
    }

    AttachingHeaders --> [*] : returns list[(chunk_text, section_header)]
```
