import json
from tqdm import tqdm
from datautils import read_jsonl
from scripts.sample_data_for_annot_situated_steps import remove_step_numbers
from datautils.markdown_cell_analysis import get_title_from_markdown, process_markdown, strip_html_tags, strip_q_tags, strip_problem_tags, strip_number_tags

# main
if __name__ == "__main__":
    val_data = read_jsonl("./data/juice-dataset/devdedup.jsonl")
    seed_queries = set()
    for rec in tqdm(val_data):
        for cell in rec["context"]:
            if cell["cell_type"] == "markdown":
                # remove html tags:
                seed_query = strip_html_tags(cell["nl_original"])
                seed_query = process_markdown(get_title_from_markdown(seed_query))
                seed_query = remove_step_numbers(seed_query)
                seed_query = seed_query.replace("(TODO)", " ")
                seed_query = seed_query.replace('"', " ").replace("'", ' ').strip()
                seed_query = seed_query.replace("(1.0 point)", " ")
                seed_query = seed_query.replace("-", " ")
                seed_query = strip_q_tags(seed_query)
                seed_query = strip_problem_tags(seed_query)
                seed_query = strip_number_tags(seed_query)
                for punct in [".", "!", '"', "'", "!", ")", "(", "[", "]", "}", "{", "Exercise"]:
                    seed_query = seed_query.strip(punct)
                # seed_query = seed_query.strip(".").strip("!").strip('"').strip("'").strip(")").strip("(").strip("[").strip("]").strip("}").strip("{")
                seed_query = " ".join(seed_query.split())
                if len(seed_query) == 0: continue
                if len(seed_query.split()) > 25: continue
                seed_queries.add(seed_query)
    seed_queries = sorted(list(seed_queries))
    # print(seed_queries[:10])
    print(len(seed_queries))
    with open("./data/juice-dataset/seed_queries.json", "w", encoding="utf8") as f:
        json.dump(seed_queries, f, indent=4, ensure_ascii=False)