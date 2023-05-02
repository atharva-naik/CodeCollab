import json
from datautils import read_jsonl
from datautils.markdown_cell_analysis import get_title_from_markdown, process_markdown

# main
if __name__ == "__main__":
    val_data = read_jsonl("./data/juice-dataset/devdedup.jsonl")
    seed_queries = set()
    for rec in val_data:
        for cell in rec["context"]:
            if cell["cell_type"] == "markdown":
                seed_queries.add(process_markdown(get_title_from_markdown(cell["nl_original"])))
    seed_queries = sorted(list(seed_queries))
    print(seed_queries[:10])
    print(len(seed_queries))
    