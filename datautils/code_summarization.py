# # code to summarize intents of code blocks.
# from datautils import read_jsonl
# from transformers import RobertaTokenizer, T5ForConditionalGeneration

# class CodeSummarizer:
#     """summarize code to NL intent"""
#     def __init__(self, model_path: str= "stmnk/codet5-small-code-summarization-python"):
#         self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
#         self.model = T5ForConditionalGeneration.from_pretrained(model_path)

#     def __call__(self, text: str, max_length: int=40):
#         # text = "def greet(user): print(f'hello <extra_id_0>!')"
#         input_ids = self.tokenizer(text, return_tensors="pt").input_ids
#         # simply generate a single sequence
#         generated_ids = self.model.generate(input_ids, max_length=max_length)

#         return self.tokenizer.decode(
#             generated_ids[0], 
#             skip_special_tokens=True,
#         )

# # main
# if __name__ == "__main__":
#     code_summarizer = CodeSummarizer()
#     val_data = read_jsonl("./data/juice-dataset/dev.jsonl")
#     code = "def greet(user): print(f'hello <extra_id_0>!')"
#     # val_data[0]["code"]
#     intent = code_summarizer(code)
#     print(code)
#     print(intent)

from transformers import RobertaTokenizer, T5ForConditionalGeneration

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    text = """def svg_to_image(string, size=None):
    if isinstance(string, unicode):
        string = string.encode('utf-8')
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
    if not renderer.isValid():
        raise ValueError('Invalid SVG data.')
    if size is None:
        size = renderer.defaultSize()
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
    return image"""

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=20)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))