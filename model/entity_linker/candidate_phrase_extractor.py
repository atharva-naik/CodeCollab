# extract candidate phrases from markdowns/codes.
import nltk
import spacy
import warnings
from datautils import read_jsonl
from nltk.corpus import stopwords
from collections import defaultdict
from datautils.markdown_cell_analysis import process_markdown

class MarkdownPhraseExtractor:
    def __init__(self, model_path: str="en_core_web_lg", kp_path: str="./analysis/dev_keyphrases.jsonl"):
        # download nltk stop words:
        nltk.download('stopwords')
        self.model = spacy.load(model_path)
        self.stop_words = stopwords.words("english")
        self.cached_kps = {}
        loaded_kps = read_jsonl(kp_path)
        for rec in loaded_kps:
            md = rec["markdown"]
            del rec["marl"]
            self.cached_kps[md].append(rec)

    def expand_context(self, span: spacy.tokens.span.Span, doc: spacy.tokens.doc.Doc, k: int=0):
        first_token = span[0]
        last_token = span[-1]
        i = max(first_token.i-k, 0)
        j = min(last_token.i+k, len(doc)-1)
        text = []
        for idx in range(i, j+1):
            text.append(doc[idx].text)

        return " ".join(text)

    def __call__(self, cell: str, k: int=0, 
                 kp_and_title_mode: bool=True):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        phrases = []
        # get the KPs for this markdown
        if kp_and_title_mode:
            first_para = para.split("\n")[0].strip()
            if first_para.startswith("#"):
                phrases.append(first_para.strip("#"))
            phrases += self.cached_kps[cell]["keyphrases"]
        else:
            for para in cell.split("\n"):
                para = para.strip("\n")
                para = process_markdown(para)
                doc = self.model(para)
                phrases += [self.expand_context(nc, doc, k=k) for nc in doc.noun_chunks if nc.text.lower() not in self.stop_words]

        return phrases

# main
if __name__ == "__main__":
    md_ext = MarkdownPhraseExtractor()
    from datautils import read_jsonl
    nbs = read_jsonl("./data/juice-dataset/dev.jsonl")
    for cell in nbs[0]["context"]:
        if cell["cell_type"] == "markdown": break
    phrases = md_ext(cell["nl_original"], k=2)
    print(cell["nl_original"])
    print(phrases)