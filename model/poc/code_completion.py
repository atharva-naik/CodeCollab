# complete partial solution written by the student.
import ast
from text_generation import Client

def remove_incomplete_code_line_by_line(code: str):
    """remove a line of code from a predicted completion till it can be parsed into an AST."""
    while True:
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            # chop off a line of code.
            code = "\n".join(code.split("\n")[:-1])

class HfCodeCompleter:
    def __init__(self, model_host_url: str="http://tir-1-32:8880", timeout: int=60):
        self.timeout = timeout
        self.client = Client(model_host_url, timeout=timeout)

    def complete(
        self, partial_soln: str, add_new_line: bool=True,
        max_new_tokens=64, top_p=0.95, temperature=0.2, 
        do_sample=True, discard_over_generations: bool=True,
    ):
        partial_soln = partial_soln.strip("\n")
        if add_new_line: partial_soln += "\n"
        gen_text = self.client.generate(
            partial_soln, max_new_tokens=max_new_tokens, 
            temperature=temperature, do_sample=do_sample,
            top_p=top_p,
        ).generated_text
        if discard_over_generations:
            gen_text = self.discard_over_generation(partial_soln, gen_text)

        return gen_text

    def discard_over_generation(self, partial_soln: str, gen_text: str):
        """discard over generated parts of solutions for specific settings like function completion."""
        complete_code = remove_incomplete_code_line_by_line(partial_soln + gen_text)
        for node in ast.walk(ast.parse(complete_code)):
            if isinstance(node, ast.FunctionDef):
                return ast.unparse(node)

# main
if __name__ == "__main__":
    ref_soln = '''def movie_count_by_genre(movies, genres):
    """
    Count the number of movies in each movie genre.

    args:
        movies (pd.DataFrame) : Dataframe containing movie attributes
        genres (List[str]) :  the list of movie genres

    return:
        Dict[str, Dict[int, int]]  : a nested mapping from movie genre to year to number of movies in that year
    """
    result = {}

    for g in genres:
      temp = movies.groupby([g,movies.release_date.str[-4:]]).agg(movie_count = ("movie_title", "count")).reset_index()
      years = temp[temp[g]==1][['release_date','movie_count']].set_index('release_date').to_dict()['movie_count']
      years = {int(k):int(v) for k,v in years.items()}
      result[g] = years

    return result
'''
    partial_soln = '''def movie_count_by_genre(movies, genres):
    """
    Count the number of movies in each movie genre.

    args:
        movies (pd.DataFrame) : Dataframe containing movie attributes
        genres (List[str]) :  the list of movie genres

    return:
        Dict[str, Dict[int, int]]  : a nested mapping from movie genre to year to number of movies in that year
    """
    result = {}

    for g in genres:'''

    hf_codegen = HfCodeCompleter()
    soln = hf_codegen.complete(partial_soln)
    print("\x1b[34;1mPartial Solution:\x1b[0m")
    print(partial_soln)
    print("\x1b[34;1mSolution:\x1b[0m")
    print(soln)