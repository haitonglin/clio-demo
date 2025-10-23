# clio-demo

### Setup
1. `conda create -n uv_clio python=3.11.9`
2. `conda activate uv_clio`
3. `pip install -r requirements.txt`
4. Set your API key:
   - macOS/Linux: `export ANTHROPIC_API_KEY="sk-ant-xxxx"`
   - Windows: `setx ANTHROPIC_API_KEY "sk-ant-xxxx"`

what each file means

main.ipynb
- where the entire pipeline will be carried out


sample_data.csv:
- geneated by gpt5o 
- 1000 sample raw chats in a csv 
- columns: chat_id, timestamp, title, category, encoded_content
- simple conversations (2-4 turns)
- no other info (no languge, metadata, etc)


preprocessing.py:
helper functions that:
- convert base64-encoded string to json string
- decode base64-encoded data
- transform into xml

chats_xml.csv
- preprocessed chats

facet_extraction.py
- using anthropic api here (my own api key)
- extract 4 facets
- slightly modification for efficiency: extract all facets using 1 call (instead of 4)
- the exact paper version (with 4 calls) is still in the paper

clustered_names.csv: csv with all clustere's names
clustered_named.csv: each row has the chat id, facets, cluster_id, the summary of the cluster it belongs to and the cluster's name