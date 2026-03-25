# Data Structure Update - Multi-Release Support

## Summary

Updated the data ingestion and indexing system to support multiple 3GPP specification releases (Rel-15, Rel-16, Rel-17) organized in separate subdirectories.

## Changes Made

### 1. Updated `discover_markdown_files()` in `data_ingestion.py`

**Before:**
- Only searched for `.md` files in a single flat directory (`data/raw/*.md`)

**After:**
- Recursively searches for `.md` files in all subdirectories (`data/**/*.md`)
- Supports both legacy flat structure and new multi-release structure

### 2. Updated `specagent index` CLI command in `cli.py`

**Before:**
- Default data directory: `data/raw`
- Used `download_38_series_specs()` for downloads

**After:**
- Default data directory: `data` (searches recursively)
- Uses `download_all_required_specs()` for downloads
- Downloads all required specs across multiple releases

## Directory Structure

```
data/
├── raw_rel_15_36_series/    # Rel-15 36-series (2 files)
│   ├── 36777-f00_1.md
│   └── 36777-f00_2.md
├── raw_rel_15_38_series/    # Rel-15 38-series (1 file)
│   └── 38811-f40.md
├── raw_rel_16_38_series/    # Rel-16 38-series (2 files)
│   ├── 38821-g20.md
│   └── 38901-g10.md
├── raw_rel_17_36_series/    # Rel-17 36-series (123 files)
│   └── [123 specification files]
├── raw_rel_17_38_series/    # Rel-17 38-series (169 files)
│   └── [169 specification files]
└── lancedb/                 # LanceDB vector store (persistent, embedded)
```

**Total: 297 specification files**

## Usage

### CLI Download & Index

```bash
# Build LanceDB index from all spec files under data/
specagent index --force

# Ingest from a custom directory
specagent index --docs-dir PATH --force
```

## Testing

Verified:
- ✓ All 297 files are discoverable
- ✓ Files are readable from all subdirectories
- ✓ Download function creates correct directory structure
- ✓ Recursive glob pattern works correctly

## Future Usage

To manually download and index specifications in the future:

1. Place spec markdown files under `data/` (any subdirectory structure).

2. **Build LanceDB index:**
   ```bash
   specagent index --force
   ```
