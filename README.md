# Clio Demo

A conversation analysis pipeline implementing the [Clio methodology](http://arxiv.org/abs/2412.13678) for extracting facets, clustering, and hierarchical organization.

## Overview

This project uses:
- **Facet extraction:** with Claude 3.5 Haiku (request, language, task, concern)
- **Embedding-based clustering:** with sentence-transformers (all-mpnet-base-v2)
- **LLM-powered cluster naming:** for interpretability
- **UMAP visualization:** for cluster inspection
- **Hierarchical organization:** for structured insights

Things further implement:
- **Full hierarchizer** would be used with more clusters
- **Privacy auditor** one final step before publishing, also would only be useful with more clusters
- **Visualizations** TBD with more detailed data + RQs
---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n clio_demo python=3.11.9
conda activate clio_demo

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

| Platform | Command |
|----------|---------|
| macOS/Linux | `export ANTHROPIC_API_KEY="sk-ant-xxxx"` |
| Windows | `setx ANTHROPIC_API_KEY "sk-ant-xxxx"` |

### 3. Run Pipeline

Open and execute `main.ipynb` to run the complete pipeline.

---

## Pipeline Stages

The `main.ipynb` notebook executes these stages in sequence:

| Stage | Description |
|-------|-------------|
| 1. Data Preprocessing | Decode base64 conversations to XML format |
| 2. Facet Extraction | Extract 4 facets using Claude |
| 3. Embedding & Clustering | Generate embeddings and perform K-means |
| 4. Cluster Naming | Generate human-readable cluster names |
| 5. Visualization | Create UMAP projections for inspection |
| 6. Hierarchy Building | Organize clusters hierarchically |

---

## Core Modules

### Module Overview

| Module | Purpose | Key Innovation |
|--------|---------|----------------|
| `preprocessing.py` | Data preprocessing & XML transformation | Clean base64 → XML pipeline |
| `facet_extraction.py` | Extract facets with Claude 3.5 Haiku | **4x faster batched mode** |
| `clustering.py` | Embedding-based clustering | Intelligent k selection |
| `cluster_naming.py` | LLM-powered cluster naming | Exact paper implementation |
| `projector.py` | UMAP 2D visualization | Cosine-metric projections |
| `hierarchizer_demo.py` | Hierarchical cluster organization | Agglomerative clustering |

---

## Module Details

### preprocessing.py

**Purpose:** Handles conversation data preprocessing and XML transformation

| Function | Description |
|----------|-------------|
| `decode_row_to_turns()` | Decodes base64-encoded conversation data to structured turns |
| `xml_transform()` | Converts conversation turns to XML format for LLM processing |

**Implementation:** Clean base64 → JSON → turns → XML pipeline as described in the Clio paper.

---

### facet_extraction.py

**Purpose:** Extracts 4 facets from conversations using Claude 3.5 Haiku

| Function | Description |
|----------|-------------|
| `run_facets_on_dataframe()` | **BATCHED MODE** (default, 4x faster) - 1 API call per conversation |
| `run_facets_on_dataframe_sequential()` | Original paper method - 4 API calls per conversation |
| `ClaudeClient` | Wrapper for Anthropic API integration |

**Key Modifications:**
- **Major optimization:** Batched facet extraction (1 API call vs 4)
- Maintains exact Clio paper prompts and safety guidelines
- Preserves original 4-call method for reproducibility
- Handles PII removal and de-identification per paper requirements

---

### clustering.py

**Purpose:** Implements embedding-based clustering using sentence transformers

| Function | Description |
|----------|-------------|
| `embed_facets()` | Generate embeddings using all-mpnet-base-v2 |
| `run_kmeans()` | K-means clustering with automatic k selection |
| `choose_k()` | Heuristic for selecting optimal number of clusters |
| `assign_clusters()` | Assign cluster labels to data |
| `summarize_clusters()` | Generate cluster size statistics |

**Key Modifications:**
- Intelligent k selection: `k = n_samples // 50`, capped at 40
- Optimized scikit-learn KMeans parameters
- Maintains paper's embedding model (all-mpnet-base-v2)

---

### cluster_naming.py

**Purpose:** Generates human-readable names and summaries for clusters using LLM

| Function | Description |
|----------|-------------|
| `name_all_clusters()` | Main function to name all clusters |
| `sample_cluster_examples()` | Paper-accurate sampling (≤50 in-cluster + 50 nearest out-of-cluster) |
| `build_cluster_prompt()` | Constructs Clio-style prompts for cluster naming |
| `generate_cluster_name()` | Single cluster naming with fallback handling |

**Key Modifications:**
- **Exact paper implementation** of sampling strategy and prompt structure
- Robust error handling with fallback name generation
- Configurable parameters for different use cases
- Saves prompts and outputs for debugging

---

### projector.py

**Purpose:** UMAP-based 2D visualization for cluster inspection

| Function | Description |
|----------|-------------|
| `compute_umap()` | Generate 2D UMAP projections |
| `project_and_merge()` | Merge projections with original data |
| `plot_projection()` | Create scatter plots for cluster visualization |

**Key Modifications:**
- Uses cosine distance metric (appropriate for sentence embeddings)
- Configurable UMAP parameters
- Caching system for reproducible results
- Clean matplotlib-based plotting (no seaborn dependency)

---

### hierarchizer_demo.py

**Purpose:** Builds hierarchical structure for organizing clusters

| Function | Description |
|----------|-------------|
| `build_hierarchy()` | Main hierarchy construction |
| `compute_cluster_centroids()` | Calculate cluster centroids |
| `group_clusters()` | Agglomerative clustering on centroids |

**Key Modifications:**
- Simplified hierarchy building for demo
- Uses cosine distance for cluster similarity
- Configurable hierarchy depth and parameters
- JSON output for easy integration

---

## Data Files

### Input Data

| File | Description |
|------|-------------|
| `sample_data.csv` | 1000 sample conversations generated by GPT-5o |

**Structure:**
- **Columns:** `id`, `created_dttm`, `title`, `tags`, `encoded_content`
- **Format:** Base64-encoded conversation data
- **Content:** Simple 2-4 turn conversations

### Generated Files

| File | Description |
|------|-------------|
| `data/chats_xml.csv` | Preprocessed conversations in XML format |
| `data/facets.csv` | Extracted facets for all conversations |
| `data/clustered_named.csv` | Final results with cluster assignments and names |
| `embeddings.npy` | Saved embeddings for reuse |
| `artifacts/` | Directory containing intermediate results and visualizations |

---

## Key Features

| Feature | Benefit |
|---------|---------|
| **Batched Processing** | 4x faster facet extraction |
| **Automatic K Selection** | Smart cluster count determination |
| **Paper-Accurate Implementation** | Faithful to Clio methodology |
| **Interactive Visualization** | UMAP projections for exploration |
| **Hierarchical Organization** | Multi-level cluster structure |
| **Reproducible Results** | Caching and saved artifacts |

---

## Technical Details

### Models Used

| Component | Model |
|-----------|-------|
| Facet Extraction | Claude 3.5 Haiku |
| Embeddings | all-mpnet-base-v2 (sentence-transformers) |
| Clustering | K-means (scikit-learn) |
| Dimensionality Reduction | UMAP |
| Hierarchy | Agglomerative Clustering |

### Facet Types

| Facet | Description |
|-------|-------------|
| Request | What the user is asking for |
| Language | The style and tone of communication |
| Task | The underlying task or goal |
| Concern | Areas of user concern or importance |