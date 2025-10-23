# Clio Demo

A Python implementation of the Clio paper's methodology for conversation analysis and clustering. This demo processes chat conversations through a complete pipeline: preprocessing → facet extraction → clustering → naming → visualization → hierarchy building.

## Overview

This project implements the Clio methodology for analyzing and clustering conversations using:
- **Facet extraction** using Claude 3.5 Haiku to identify request, language, task, and concern facets
- **Embedding-based clustering** using sentence-transformers (all-mpnet-base-v2)
- **Cluster naming** using LLM-generated summaries and names
- **Visualization** using UMAP projection
- **Hierarchy building** for organizing clusters

## Setup

1. Create conda environment:
   ```bash
   conda create -n clio_demo python=3.11.9
   conda activate clio_demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Anthropic API key:
   - macOS/Linux: `export ANTHROPIC_API_KEY="sk-ant-xxxx"`
   - Windows: `setx ANTHROPIC_API_KEY "sk-ant-xxxx"`

## Project Structure

### Core Pipeline Files

#### `main.ipynb`
The main execution notebook containing the complete Clio pipeline:
1. **Data Preprocessing** - Decode base64 conversations to XML format
2. **Facet Extraction** - Extract 4 facets (request, language, task, concerning) using Claude
3. **Embedding & Clustering** - Generate embeddings and perform K-means clustering
4. **Cluster Naming** - Generate human-readable names and summaries for clusters
5. **Visualization** - UMAP projection for cluster inspection
6. **Hierarchy Building** - Organize clusters into hierarchical structure

### Python Modules

#### `preprocessing.py`
**Purpose**: Handles conversation data preprocessing and XML transformation.

**Key Functions**:
- `decode_row_to_turns()` - Decodes base64-encoded conversation data to structured turns
- `xml_transform()` - Converts conversation turns to XML format for LLM processing

**Modifications**: Clean implementation of the base64 → JSON → turns → XML pipeline as described in the Clio paper.

#### `facet_extraction.py`
**Purpose**: Extracts 4 facets from conversations using Claude 3.5 Haiku.

**Key Functions**:
- `run_facets_on_dataframe()` - **BATCHED MODE** (default, 4x faster)
- `run_facets_on_dataframe_sequential()` - Original paper method (4 API calls per conversation)
- `ClaudeClient` - Wrapper for Anthropic API integration

**Modifications**: 
- **Major optimization**: Implemented batched facet extraction (1 API call per conversation instead of 4)
- Maintains exact Clio paper prompts and safety guidelines
- Preserves original 4-call method for reproducibility
- Handles PII removal and de-identification as per paper requirements

#### `clustering.py`
**Purpose**: Implements embedding-based clustering using sentence transformers.

**Key Functions**:
- `embed_facets()` - Generate embeddings using all-mpnet-base-v2
- `run_kmeans()` - K-means clustering with automatic k selection
- `choose_k()` - Heuristic for selecting optimal number of clusters
- `assign_clusters()` - Assign cluster labels to data
- `summarize_clusters()` - Generate cluster size statistics

**Modifications**: 
- Added intelligent k selection based on dataset size (heuristic: k = n_samples // 50, capped at 40)
- Uses scikit-learn's KMeans with optimized parameters
- Maintains paper's embedding model choice (all-mpnet-base-v2)

#### `cluster_naming.py`
**Purpose**: Generates human-readable names and summaries for clusters using LLM.

**Key Functions**:
- `name_all_clusters()` - Main function to name all clusters
- `sample_cluster_examples()` - Paper-accurate sampling (≤50 in-cluster + 50 nearest out-of-cluster)
- `build_cluster_prompt()` - Constructs Clio-style prompts for cluster naming
- `generate_cluster_name()` - Single cluster naming with fallback handling

**Modifications**:
- **Exact paper implementation** of sampling strategy and prompt structure
- Added robust error handling with fallback name generation
- Configurable parameters for different use cases
- Saves prompts and outputs for debugging and analysis

#### `projector.py`
**Purpose**: UMAP-based 2D visualization for cluster inspection.

**Key Functions**:
- `compute_umap()` - Generate 2D UMAP projections
- `project_and_merge()` - Merge projections with original data
- `plot_projection()` - Create scatter plots for cluster visualization

**Modifications**:
- Uses cosine distance metric (appropriate for sentence embeddings)
- Configurable UMAP parameters for different visualization needs
- Caching system for reproducible results
- Clean matplotlib-based plotting (no seaborn dependency)

#### `hierarchizer_demo.py`
**Purpose**: Builds hierarchical structure for organizing clusters.

**Key Functions**:
- `build_hierarchy()` - Main hierarchy construction
- `compute_cluster_centroids()` - Calculate cluster centroids
- `group_clusters()` - Agglomerative clustering on centroids

**Modifications**:
- Simplified hierarchy building for demo purposes
- Uses cosine distance for cluster similarity
- Configurable hierarchy depth and parameters
- JSON output for easy integration with other tools

### Data Files

#### `sample_data.csv`
- 1000 sample conversations generated by GPT-5o
- Columns: `id`, `created_dttm`, `title`, `tags`, `encoded_content`
- Simple 2-4 turn conversations
- Base64-encoded conversation data

#### Generated Files
- `data/chats_xml.csv` - Preprocessed conversations in XML format
- `data/facets.csv` - Extracted facets for all conversations
- `data/clustered_named.csv` - Final results with cluster assignments and names
- `embeddings.npy` - Saved embeddings for reuse
- `artifacts/` - Directory containing intermediate results and visualizations

## Key Modifications from Original Paper

### 1. **Batched Facet Extraction** (Major Optimization)
- **Original**: 4 separate API calls per conversation
- **Modified**: 1 API call per conversation (4x speed improvement)
- **Impact**: Dramatically reduces processing time and API costs
- **Preservation**: Original 4-call method still available for reproducibility

### 2. **Intelligent K Selection**
- **Original**: Manual k selection
- **Modified**: Automatic k selection based on dataset size
- **Formula**: k = min(max(3, n_samples // 50), 40)
- **Benefit**: Reduces manual parameter tuning

### 3. **Enhanced Error Handling**
- **Original**: Basic error handling
- **Modified**: Robust fallback mechanisms for LLM failures
- **Features**: Automatic retry, fallback name generation, detailed logging

### 4. **Improved Visualization**
- **Original**: Basic plotting
- **Modified**: Professional UMAP visualizations with caching
- **Features**: Reproducible results, configurable parameters, clean aesthetics

### 5. **Modular Architecture**
- **Original**: Monolithic implementation
- **Modified**: Clean, modular Python package structure
- **Benefits**: Easy to extend, test, and maintain

## Usage

1. **Run the complete pipeline**:
   ```bash
   jupyter notebook main.ipynb
   ```

2. **Use individual modules**:
   ```python
   from preprocessing import decode_row_to_turns, xml_transform
   from facet_extraction import run_facets_on_dataframe, ClaudeClient
   from clustering import embed_facets, run_kmeans
   # ... etc
   ```

## Requirements

See `requirements.txt` for complete dependency list. Key dependencies:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Clustering algorithms
- `sentence-transformers` - Text embeddings
- `umap-learn` - Dimensionality reduction
- `matplotlib` - Visualization
- `anthropic` - Claude API integration
- `jupyter` - Notebook environment

## Notes

- **API Costs**: Facet extraction is the most expensive step (uses Claude API)
- **Processing Time**: Full pipeline takes ~5-10 minutes for 100 conversations
- **Scalability**: Designed for datasets up to 1000 conversations (local processing)
- **Reproducibility**: All random seeds are set for consistent results
- **Privacy**: PII removal happens during facet extraction step

## Future Improvements

- [ ] Switch to in-house LLM for facet extraction
- [ ] Implement parallel processing for large datasets
- [ ] Add more sophisticated hierarchy building
- [ ] Create web-based visualization interface
- [ ] Add evaluation metrics for cluster quality