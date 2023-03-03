# code to summarize intents of code blocks.
from datautils import read_jsonl
from transformers import RobertaTokenizer, T5ForConditionalGeneration

class CodeSummarizer:
    """summarize code to NL intent"""
    def __init__(self, model_path: str= "stmnk/codet5-small-code-summarization-python"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def __call__(self, text: str, max_length: int=10):
        # text = "def greet(user): print(f'hello <extra_id_0>!')"
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        # simply generate a single sequence
        generated_ids = self.model.generate(input_ids, max_length=max_length)

        return self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True,
        )

# main
if __name__ == "__main__":
    code_summarizer = CodeSummarizer()
    val_data = read_jsonl("./data/juice-dataset/dev.jsonl")
    code = val_data[0]["code"]
    intent = code_summarizer(code)
    print(code)
    print(intent)