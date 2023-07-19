import json
from typing import *

curated_ops = {
    "1. Series and Dataframes": {
        "1.1. Series": [
            's = pd.Series({"a" : 10, "b" : 20, "c" : 30})',
            's = pd.Series(x**2 for x in range(5, 10))',
        ],
        "1.2. Dataframes": [
            'pd.DataFrame([[1, 2], [3, 4], [5, 6]])',
            'pd.DataFrame([{"a" : 1}, {"a" : 2, "b" : 3}, {"c" : 4}])',
            'pd.DataFrame({"a" : [1, 3, 5, 7], "b" : range(4), "c" : np.ones(4)})',
            'df = pd.DataFrame({"a" : [1, 3, 5, 7], "b" : range(4), "c" : np.ones(4)}, index = ["x", "y", "z", "h"])',
            "pd.read_csv('example.csv')",
            "pd.read_csv('example.csv', index_col=False)",
            "pd.read_csv('example.tsv', delimiter='\t')",
            "pd.read_csv('example.tsv', dtype= {'a': np.float64, 'b': np.int32, 'c': 'Int64'})",
        ],
        "1.3. Data Access": [
            'df.iloc[2,1]',
            'df.iloc[0, [0,2]]',
            'df.iloc[[0], [0, 2]]',
            'df.iloc[1]',
            'df.iloc[:,1]',
            'df.iloc[1:,:-1]',
            'df.loc["x", "c"]',
            'df.loc["x", ["a", "c"]]',
            'df.loc[["x"], ["a", "c"]]',
            'df.loc["y"]',
            'df.loc["y":, :"c"]',
            'df.loc[[True, False, True, False], [False, True, False]]',
            'df["a"]',
            'df[["b, "c"]])',
            'df["x":"h"]',
            'df[[True, False, False, True]]',
            's["b"]',
            's[["a", "c"]]',
            's[[True, True, False]]',
            's.iloc[[1, 2]]'
        ],
        "1.4. Dealing with SettingWithCopyWarning": [
            """column = df["a"]
    column[0] = 100""",
            """df.loc[0, "a"] = 100""",
            """column = df["a"].copy()
    column[0] = 100""",
        ],
        "1.5. Pandas and NumPy": {
            "convert NumPy matrix to Pandas dataframe": ["df = pd.DataFrame(np.arange(5000).reshape(1000, 5))", "s = pd.Series(np.random.randint(low = 0, high = 10, size = 1000))", """import string
# series of random alphabet letters
np.random.seed(0)
s = pd.Series(np.random.choice(list(string.ascii_letters), size = 1000))""", 'df = pd.DataFrame(np.random.randint(0, 10, size = 5000).reshape(1000, 5))'],
            "convert Pandas dataframe to Numpy matrix": ["df.values"],
        },
    },
    "2.1. Series iteration": {
        "elementwise operation between multiple series": ["s + s/2 - s**2"],
        "frequency count for each unique value": ["s.value_counts()"],
        "data overview": ["s.describe()"],
        "standard numerical operations": ["s.sum()", "s.std()", "s.mean()"],
        "extract unique values": ["s.unique()", "s.nunique()"],
        "convert to lower case": ["s.str.lower()"],
        "get string length": ["s.str.len()"],
        "lowercase strings and replace": ["s.str.lower().str.replace('s', '*')", "s.apply(lambda x: x.lower().replace('s', '*'))"],   
    },
    "2.2 DataFrame iteration": {
        "dataframe iteration": [
            "for col in df.columns:",
            "for index, row in df.iterrows():"
        ],
        "Sum of every column in the dataframe": ['df.sum(axis = 0)'],
        "Sum of every row in the dataframe": ['df.sum(axis = 1)'],
        "data filtering": [
            "df.loc[:, (df%2 == 1).sum(axis = 0) > len(df)/2]",
            "df[df.sum(axis = 1) % 3 == 0]"
        ],
    }
}

def unfold_dict_recursively(d: dict, subpath: List[str]=[]) -> Dict[str, Dict[str, Union[List[str], str]]]:
    """Recursively unfold a dictionary to get mapping from lowest level names
    to code instantiations and expanded paths."""
    ops = {}
    for k, v in d.items():
        if isinstance(v, list):
            ops[k] = {
                "codes": v,
                "path": '/'.join(subpath+[k])
            }
        elif isinstance(v, dict):
            ops.update(unfold_dict_recursively(v, subpath=subpath+[k]))

    return ops

# main
if __name__ == "__main__":
    unfolded_curated_ops = unfold_dict_recursively(curated_ops, subpath=[])
    with open("./data/FCDS/primer_only_plan_ops.json", "w") as f:
        json.dump(unfolded_curated_ops, f, indent=4)