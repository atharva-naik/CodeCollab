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
    codes = [
        """
for n_neighbors in [1, 5, 10, 20, 30]:
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print(n_neighbors, knn.score(X_test, y_test))
""",
"""config_logging('INFO')
possible = IterChannel(KeyedArtifact(i, DerivedArtifact(expensive_deriver, i, name='expensive')) for i in range(10))
extant = IterChannel(KeyedArtifact(i, ExampleExtantArtifact(i)) for i in keys)
all_ = merge_keyed_channels(possible, extant)
print_chans(all_.tee())
config_logging('WARN')""",
"""m= ipyl.Map(scroll_wheel_zoom=True, center=[-0.3515602939922709, 22.5], 
            zoom=1, layout=ipyw.Layout(width='45%', height='450px'))

dc = ipyl.DrawControl(polygon={'shapeOptions': {'color': 'green', 'weight': 2, 'clickable': True}})
m.add_control(dc)

for centroid in df.geometry:
    m.add_layer(ipyl.GeoJSON(data=mapping(centroid)))""",
"""from ipyleaflet import (
    Map, Marker, TileLayer, ImageOverlay,
    Polyline, Polygon, Rectangle, Circle,
    CircleMarker, GeoJSON, DrawControl,
    FeatureGroup
)
hi_bbox = {
    "type": "Polygon",
    "coordinates": [
        [
            [
              -171.03515625,
              6.926426847059551
            ],
            [
              -171.03515625,
              33.797408767572485
            ],
            [
              -144.580078125,
              33.797408767572485
            ],
            [
              -144.580078125,
              6.926426847059551
            ],
            [
              -171.03515625,
              6.926426847059551
            ]
        ]
    ]
}
center = [20, -157]
m = Map(center=center, zoom=4)
g = GeoJSON(data=hi_bbox)
m.add_layer(g)
m"""]
    codes = [code.strip("\n") for code in codes]
    # text = """def svg_to_image(string, size=None):
    # if isinstance(string, unicode):
    #     string = string.encode('utf-8')
    #     renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
    # if not renderer.isValid():
    #     raise ValueError('Invalid SVG data.')
    # if size is None:
    #     size = renderer.defaultSize()
    #     image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
    #     painter = QtGui.QPainter(image)
    #     renderer.render(painter)
    # return image"""
    for code in codes:
        input_ids = tokenizer(code, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=20)
        print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))