import re
import json
from tqdm import tqdm
from datautils import read_jsonl
from collections import defaultdict
from scripts.sample_data_for_annot_situated_steps import remove_step_numbers
from datautils.markdown_cell_analysis import get_title_from_markdown, process_markdown, strip_html_tags, strip_q_tags, strip_problem_tags, strip_number_tags

def process_query(q: str):
    # remove html tags:
    seed_query = strip_html_tags(q)
    # remove (x points) patterns.
    seed_query = re.sub(r'\(\d+ points\)', '', seed_query)
    # print(f"11: {seed_query}")
    seed_query = process_markdown(get_title_from_markdown(seed_query))
    # print(f"13: {seed_query}")
    seed_query = remove_step_numbers(seed_query)
    # print(f"15: {seed_query}")
    seed_query = seed_query.replace("(TODO)", " ")
    # print(f"17: {seed_query}")
    seed_query = seed_query.replace('"', " ").replace("'", ' ').strip()
    # print(f"19: {seed_query}")
    seed_query = seed_query.replace("(1.0 point)", " ")
    # print(f"21: {seed_query}")
    seed_query = seed_query.replace("-", " ")
    # print(f"23: {seed_query}")
    seed_query = strip_q_tags(seed_query)
    # print(f"25: {seed_query}")
    seed_query = strip_problem_tags(seed_query)
    # print(f"27: {seed_query}") 
    seed_query = strip_number_tags(seed_query)
    # print(f"29: {seed_query}")
    for punct in [".", "!", "=", "–", '"', "\\", "$", "|", "'", ":", ")", "(", "[", "]", "}", "{"]:#, "Exercise"]:
        seed_query = seed_query.replace(punct, " ")
    seed_query = seed_query.strip()
    if seed_query.startswith("COGS "):
        seed_query = seed_query[len("COGS "):]
    seed_query = seed_query.strip()
    if seed_query.startswith("Exercise "):
        seed_query = seed_query[len("Exercise "):]
    seed_query = seed_query.strip()
    if seed_query.startswith("Aside "):
        seed_query = seed_query[len("Aside "):]
    seed_query = seed_query.strip()
    if seed_query.startswith("Week "):
        seed_query = seed_query[len("Week "):]
    seed_query = seed_query.strip()
    if seed_query.startswith("Assignment "):
        seed_query = seed_query[len("Assignment "):]
    seed_query = seed_query.strip()
    if seed_query.startswith("Assignments "):
        seed_query = seed_query[len("Assignments "):]
    seed_query = seed_query.strip()
    if seed_query.startswith("Sheet "):
        seed_query = seed_query[len("Sheet "):]
    seed_query = seed_query.strip()
    # seed_query = seed_query.strip(".").strip("!").strip('"').strip("'").strip(")").strip("(").strip("[").strip("]").strip("}").strip("{")
    # print(f"33: {seed_query}")
    seed_query = " ".join(seed_query.split())
    seed_query = seed_query.lower()

    return seed_query

# main
if __name__ == "__main__":
    val_data = read_jsonl("./data/juice-dataset/devdedup.jsonl")
    seed_queries = defaultdict(lambda:[])
    for rec in tqdm(val_data):
        for cell in rec["context"]:
            if cell["cell_type"] == "markdown":
                original_nl = cell["nl_original"]
                seed_query = process_query(cell["nl_original"])
                if len(seed_query) == 0: continue
                if len(seed_query.split()) > 25: continue
                # seed_queries.add(seed_query)
                seed_queries[seed_query].append(original_nl)
    seed_queries = {k: v for k,v in sorted(seed_queries.items(), reverse=False, key=lambda x: x[0])}
    # print(seed_queries[:10])
    print(len(seed_queries))
    with open("./data/juice-dataset/seed_queries.json", "w", encoding="utf8") as f:
        json.dump(seed_queries, f, indent=4, ensure_ascii=False)