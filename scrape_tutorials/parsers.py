#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# parsers for various libraries/modules.
import os
import re
import ast
import bs4
import json
import bs2json
import requests
import html_to_json
from typing import *
from tqdm import tqdm
from bs2json import BS2Json
from urllib.parse import urldefrag
from collections import defaultdict
from scrape_tutorials import SOURCE_TO_BASE_URLS
from datautils.markdown_cell_analysis import extract_notebook_hierarchy_from_seq

def simplify_soup(soup, target: str="seaborn"):
    if target == "pandas":
        soup = soup.html.body.find("div", {"class": "post-content"})
    elif target == "torch":
        soup = soup.html.body.find("div", {"class": "main-content"})
    elif target == "realpython":
        pass
    elif target == "scipy":
        soup = soup.html.body.select("article.bd-article")[0]
    elif target == "scikit":
        soup = soup.html.body.select("div.sk-page-content.container-fluid.body.px-md-3")[0]
    elif target == "matplotlib":
        soup = soup.html.body.select("section.sphx-glr-example-title")[0]
        # if not try_soup:
            # try_soup = soup.html.body.find("div", {"class": ""})
        # soup = try_soup
        # assert soup is not None, f"\x1b[32;1mmain-content\x1b[0m not found"
    for aside in soup.select("aside"): aside.unwrap()
    for hr in soup.select("hr"): hr.unwrap()
    for nav in soup.select("nav"): nav.unwrap()
    for span in soup.select('span'): span.unwrap()
    for code in soup.select('code'): code.unwrap()
    for div in soup.select('div'): div.unwrap()
    for li in soup.select('li'): li.unwrap()
    for ol in soup.select('ol'): ol.unwrap()
    for ul in soup.select('ul'): ul.unwrap()
    for table in soup.select('table'): table.extract()
    for style in soup.select('style'): style.extract()
    for section in soup.select("section"): section.unwrap()
    if target not in ["scipy", "matplotlib"]:
        for a in soup.select('a'): a.extract()
    else:
        for a in soup.select('a'): a.unwrap()
    for pre in soup.select("pre"): del pre.attrs
    for p in soup.select("p"): del p.attrs
    for i in range(1, 12):
        for hi in soup.select(f"h{i}"): del hi.attrs
    # for section in soup.select("section"): section.unwrap()
    if target == "seaborn": 
        article = soup.html.body.find("article", {"class": "bd-article"})
        for content in article.contents:
            if content != "\n":
                final = content
                break
    elif target == "pandas": final = soup
    elif target == "numpy":
        final = soup.html.body.find("article", {"class": "bd-article"})
    elif target == "torch": final = soup
    elif target == "realpython": final = soup
    elif target == "scipy": final = soup
    elif target == "scikit": final = soup
    elif target == "matplotlib": final = soup

    return final

def collapse_list_of_strings(d: Union[dict, str, list]):
    if isinstance(d, str): return d
    elif isinstance(d, list):
        all_ele_are_str = True
        for v in d:
            if not isinstance(v, str):
                all_ele_are_str = False
                break
        if all_ele_are_str: new_d = "\n".join(d)
        else:
            new_d = []
            for v in d: new_d.append(collapse_list_of_strings(v))
        return new_d
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = collapse_list_of_strings(v)

    return d
# def simplify_html_to_json(html_to_json_dict: dict):
#     simple_json = {}
#     # print("simplifying JSON")
#     if isinstance(html_to_json_dict, dict):
#         if len(html_to_json_dict) == 1 and isinstance(list(html_to_json_dict.values())[0], str):
#             return list(html_to_json_dict.values())[0]
#         for key, value in html_to_json_dict.items():
#             if key == "_value": key = "content"
#             elif key == "h1": key = "markdown1"
#             elif key == "h2": key = "markdown2"
#             elif key == "h3": key = "markdown3"
#             elif key == "_values": key = "contents"
#             elif key == 'pre': key = "code"
#             # print(key)
#             if isinstance(value, (dict, list)):
#                 simple_json[key] = simplify_html_to_json(value)
#             else: simple_json[key] = value
#     elif isinstance(html_to_json_dict, list):
#         simple_json = []
#         for value in html_to_json_dict:
#             simple_json.append(simplify_html_to_json(value))
#         return simple_json
#     else: return html_to_json_dict

#     return simple_json

# def simplify_bs2_json(html_json: str, filt_keys: List[str]=[
#                       'script', 'style', 'role', 'br', 'class']):
#     if isinstance(html_json, dict):
#         if [k for k in html_json.keys() if k not in filt_keys] == ["text"]:
#             if isinstance(html_json["text"], list):
#                 return " ".join(html_json["text"])    
#             return html_json["text"]
#         new_json = {}
#         for key, value in html_json.items():
#             if key in filt_keys: 
#                 # print(key)
#                 pass
#             elif key == "attrs":
#                 for k,v in html_json["attrs"].items():
#                     if k in filt_keys: continue
#                     new_json[k] = simplify_bs2_json(v)
#             elif isinstance(value, (dict, list)):
#                 new_json[key] = simplify_bs2_json(value, filt_keys)
#             elif key == "p" and isinstance(value, list):
#                 new_json[key] = "\n".join(value)
#             elif key == "text" and isinstance(value, list):
#                 new_json[key] = " ".join(value)
#             else: new_json[key] = value
#     elif isinstance(html_json, list):
#         new_json = []
#         for value in html_json:
#             new_json.append(simplify_bs2_json(value, filt_keys))
#     else:
#         # print(type(html_json))
#         new_json = html_json

#     return new_json
def parse_soup_stream(soup_str: str, tag_mapping: Dict[str, Tuple[str, str]]={
        "h1": ("markdown", 1), "h2": ("markdown", 2), "h3": ("markdown", 3),
        "h4": ("markdown", 4), "h5": ("markdown", 5), "h6": ("markdown", 6),
        "h7": ("markdown", 7), "h8": ("markdown", 8), "h9": ("markdown", 9),
        "h10": ("markdown", 10), "h11": ("markdown", 11), "h12": ("markdown", 12),
        "p": ("markdown", 0), "pre": ("code", 0), "img": ("markdown", 0),
        "a": ("markdown", 0),
    }, default_cell: Tuple[str, str]=("markdown", 0)) -> List[dict]:
    i = 0
    cells = []
    for tag in tag_mapping: soup_str = soup_str.replace("</"+tag+">", "") # print(soup_str)
    soup_str = soup_str.strip()
    while i < len(soup_str):
        char = soup_str[i]
        found_tag = False
        for tag, tag_props in tag_mapping.items():
            if char == "<" and soup_str[i:i+len(tag)+2] == "<"+tag+">":
                # if tag == "a": cells.append({"cell_type": "markdown", "content": })
                cells.append({"cell_type": tag_props[0], "content": "#"*tag_props[1]})
                i += (len(tag)+1)
                found_tag = True
                break
        if not found_tag:
            cells[-1]["content"] += char
        i += 1
    for cell in cells:
        if cell["cell_type"] == 'markdown':
            cell["nl_original"] = cell["content"].strip("\n")
        elif cell["cell_type"] == "code":
            cell["code"] = cell["content"].strip("\n")
        del cell["content"]
    
    return cells

def process_text(text: str, cell_type: str="markdown"):
    """remove residual html tags, &amp; etc. 
    e.g.: `Reshaping &amp; Tidy Data<blockquote>` to `Reshaping & Tidy Data`"""
    text = text.replace("&amp;", "&").replace("<blockquote>", "").replace("<cite>","").replace("</cite>","").replace("<em>","").replace("</em>","").replace("<dt>","").replace("</dt>","").replace("<dd>","").replace("</dd>","")
    text = re.sub("<dl.*?>", "", text).split("\n")[0].strip()
    if cell_type == "markdown":
        text = text.replace("&gt;", ">")
    elif cell_type == "code":
        text = text.replace("&gt;", "")
    text = text.strip("#") # remove trailing (or beginning # signs).

    return text.strip()

def process_cell_text(text: str, cell_type: str="markdown"):
    """remove residual html tags, &amp; etc. 
    e.g.: `Reshaping &amp; Tidy Data<blockquote>` to `Reshaping & Tidy Data`"""
    text = text.replace("&amp;", "&").replace("<blockquote>", "").replace("<cite>","").replace("</cite>","").replace("<em>","").replace("</em>","").replace("<dt>","").replace("</dt>","").replace("<dd>","").replace("</dd>","")
    text = re.sub("<dl.*?>", "", text)#.split("\n")[0].strip()
    if cell_type == "markdown":
        text = text.replace("&gt;", ">")
    elif cell_type == "code":
        text = text.replace("&gt;", "")

    return text.strip()

# parse scikit tutorials.
class SciKitLearnParser:
    def __init__(self, base_urls: Dict[str, Dict[str, str]]=SOURCE_TO_BASE_URLS["scikit"]) -> None:
        self.base_urls = base_urls
        assert "Examples" in self.base_urls
        assert "Tutorials" in self.base_urls
        self.examples_page_url = self.base_urls["Examples"]
        self.examples_page_save_path = "./scrape_tutorials/scikit_examples.json"
        self.example_urls = {}
        self.graph = {"Examples": {}, "Tutorials": {}}
        if os.path.exists(self.examples_page_save_path):
            self.example_urls = json.load(open(self.examples_page_save_path))
        else:
            domain_dir = "https://scikit-learn.org/stable/auto_examples/"
            self.examples_page_soup = bs4.BeautifulSoup(
                requests.get(self.examples_page_url).text,
                features="lxml",
            )
            examples_section = self.examples_page_soup.find("section", id="examples")
            for section in tqdm(examples_section.select("section")):
                h2 = section.select("h2")[0].text.replace("¶","")
                p = section.select("p")[0].text
                self.example_urls[h2] = {
                    "description": p,
                    "urls": {},
                }
                for div in section.select("div.sphx-glr-thumbcontainer"):
                    key = div.select("div.sphx-glr-thumbnail-title")[0].text
                    value = div.select("a.reference.internal")[0].attrs["href"]
                    self.example_urls[h2]["urls"][key] = os.path.join(domain_dir, value)
            with open(self.examples_page_save_path, "w", encoding="utf-8") as f:
                json.dump(self.example_urls, f, indent=4, ensure_ascii=False)
        for eg_name, eg_urls in self.example_urls.items():
            self.graph["Examples"][eg_name] = []
            self.graph["Examples"][eg_name].append(eg_urls["description"])
            self.graph["Examples"][eg_name].append({})
            for name, url in eg_urls["urls"].items():
                self.graph["Examples"][eg_name][1][name] = {} 
        for name, url in self.base_urls["Tutorials"].items():
            self.graph["Tutorials"][name] = {}

    def download_tutorials(self):
        for name, url in tqdm(self.base_urls["Tutorials"].items()):
            simple_soup = simplify_soup(
                bs4.BeautifulSoup(
                    requests.get(url).text,
                    features="lxml",
                ), target="scikit",
            )
            simplified_soup = str(simple_soup)
            simplified_soup = re.sub("<div.*?>", "", simplified_soup)
            simplified_soup = re.sub("</div>", "", simplified_soup)
            simplified_soup = re.sub("<article.*?>", "", simplified_soup)
            simplified_soup = re.sub("</article>", "", simplified_soup)
            try: 
                nb_json = extract_notebook_hierarchy_from_seq(
                    parse_soup_stream(simplified_soup)
                )[0].serialize2()[""]
                # self.graph["Tutorials"][name] = nb_json
                for ele in nb_json:
                    if isinstance(ele, dict) and len(ele) == 1:
                        key = list(ele.keys())[0]
                        if key == name: value = ele[key]; break
                self.graph["Tutorials"][name] = value
            except Exception as e: print(e)

    def download_examples(self):
        for eg_name, eg_urls in self.example_urls.items():
            for name, url in tqdm(eg_urls["urls"].items()):
                simple_soup = simplify_soup(
                    bs4.BeautifulSoup(
                        requests.get(url).text,
                        features="lxml",
                    ), target="scikit",
                )
                simplified_soup = str(simple_soup)
                simplified_soup = re.sub("<div.*?>", "", simplified_soup)
                simplified_soup = re.sub("</div>", "", simplified_soup)
                simplified_soup = re.sub("<article.*?>", "", simplified_soup)
                simplified_soup = re.sub("</article>", "", simplified_soup)
                # print(simplified_soup)
                try:
                    nb_json = extract_notebook_hierarchy_from_seq(
                        parse_soup_stream(simplified_soup)
                    )[0].serialize2()[""]
                    for ele in nb_json:
                        if isinstance(ele, dict) and len(ele) == 1:
                            key = list(ele.keys())[0]
                            if key == name: value = ele[key]; break
                    self.graph["Examples"][eg_name][1][name] = value
                except Exception as e: print(e)

# parse SciPy tutorials.
class SciPyParser:
    def __init__(self, base_urls: Dict[str, str]=SOURCE_TO_BASE_URLS['scipy']):
        self.base_urls = base_urls
        self.base_soups = {}
        self.base_pages = {}

    def download(self):
        for name, url in tqdm(self.base_urls.items()):
            self.base_soups[name] = simplify_soup(
                bs4.BeautifulSoup(
                    requests.get(url).text,
                    features="lxml",
                ),
                target="scipy",
            )
            simplified_soup = str(self.base_soups[name])
            simplified_soup = re.sub("<div.*?>", "", simplified_soup)
            simplified_soup = re.sub("</div>", "", simplified_soup)
            simplified_soup = re.sub("<article.*?>", "", simplified_soup)
            simplified_soup = re.sub("</article>", "", simplified_soup)
            try: 
                nb_json = extract_notebook_hierarchy_from_seq(
                    parse_soup_stream(simplified_soup)
                )[0].serialize2()[""][0]
                assert len(nb_json) == 1 and isinstance(nb_json, dict)
                for key, value in nb_json.items(): break
                self.base_pages[key] = value
            except Exception as e:
                print(e)

# gather RealPython resources:
class RealPythonScraper:
    def __init__(self, base_urls: Dict[str, Dict[str, str]]=SOURCE_TO_BASE_URLS["realpython"]):
        self.base_urls = base_urls
        self.learning_paths = base_urls["Python Learning Paths"]
        self.learning_paths_graph = {}
        self.learning_path_soups = {}
        # scrape learning path base pages/topic urls.
        self.learning_paths_save_path = "./scrape_tutorials/realpython_learning_paths.json"
        self.learning_paths_with_courses_path = "./scrape_tutorials/realpython_learning_paths_with_courses.json"
        self.learning_paths_all_path = "./scrape_tutorials/realpython_learning_paths_all.json" # excludes interactive quizzes.
        if os.path.exists(self.learning_paths_save_path):
            self.learning_paths_graph = json.load(open(self.learning_paths_save_path))
        else:
            for name, url in tqdm(self.learning_paths.items()):
                soup = bs4.BeautifulSoup(requests.get(url).text, features="lxml")
                # body = soup.find("p", {"class": "article-body page-body"})
                self.learning_path_soups[name] = soup 
                skills = soup.select("p", class_="text-center.text-muted.mt-0.mb-3")[0].text
                skills = skills.split("Skills:")[-1].strip()
                skills = [skill.strip() for skill in skills.split(",")]
                resource_divs = soup.select("div.container.border.rounded")
                resources = [
                    {
                        "title": div.select("a.stretched-link")[1].text,
                        "type": div.select("p.small.text-muted.mb-0")[0].text, 
                        "description": div.select("p.small.text-muted.mb-0")[1].text,
                        "url": f'https://realpython.com{div.select("a.stretched-link")[0].attrs["href"]}',
                    } for div in resource_divs]
                self.learning_paths_graph[name] = {
                    "skills": skills,
                    "resources": resources,
                } 
            with open(self.learning_paths_save_path, "w") as f:
                json.dump(self.learning_paths_graph, f, indent=4)
        if os.path.exists(self.learning_paths_with_courses_path):
            self.learning_paths_graph = json.load(open(self.learning_paths_with_courses_path))
        else:
            # iterate over learning paths.
            for name, lp_resources in self.learning_paths_graph.items():
                # iterate over resource blocks:
                for i, rblock in tqdm(enumerate(lp_resources["resources"]), 
                                      total=len(lp_resources["resources"]), 
                                      desc=f"lpath:{name}"):
                    # download and parse course decompositions.
                    if rblock["type"] == "Course":
                        course_url = rblock["url"]
                        self.learning_paths_graph[name]["resources"][i]["decomposition"] = self.scrape_course(course_url)
            with open(self.learning_paths_with_courses_path, "w") as f:
                json.dump(self.learning_paths_graph, f, indent=4)
        if os.path.exists(self.learning_paths_all_path):
            self.learning_paths_graph = json.load(open(self.learning_paths_all_path))
        else:
            # iterate over learning paths.
            for name, lp_resources in self.learning_paths_graph.items():
                # iterate over resource blocks:
                for i, rblock in tqdm(enumerate(lp_resources["resources"]), 
                                      total=len(lp_resources["resources"]), 
                                      desc=f"lpath:{name}"):
                    if rblock["type"] == "Tutorial":
                        tut_url = rblock["url"]
                        self.learning_paths_graph[name]["resources"][i]["content"] = self.scrape_tutorial(tut_url)
            with open(self.learning_paths_all_path, "w") as f:
                json.dump(self.learning_paths_graph, f, indent=4)
                
    def scrape_tutorial(self, url: str):
        simple_soup = simplify_soup(bs4.BeautifulSoup(
            requests.get(url).text,
            features="lxml",
        ), target="torch")

        simplified_soup = str(simple_soup)
        simplified_soup = re.sub("<div.*?>", "", simplified_soup)
        simplified_soup = re.sub("</div>", "", simplified_soup)
        simplified_soup = re.sub("<article.*?>", "", simplified_soup)
        simplified_soup = re.sub("</article>", "", simplified_soup)
        try: 
            nb_json = extract_notebook_hierarchy_from_seq(
                parse_soup_stream(simplified_soup)
            )[0].serialize2()[""]
            return nb_json
        except Exception as e:
            print(e)
            return {}

    def scrape_course(self, url: str):
        course_decomp = {}
        try: course_soup = bs4.BeautifulSoup(requests.get(url).text, features="lxml")
        except ConnectionError:
            print("\x1b[31;1mERROR for:\x1b[0m", url)
            return {}
        # iterate over course blocks:
        for cblock in course_soup.select("div.mb-5.p-3.bg-white.rounded.border.shadow-sm"):
            key = cblock.select("h2.border-bottom.border-gray.pb-2.mb-0.h3.mt-0.pt-0")[0].text
            value = [p.text.strip() for p in cblock.select("p.h5.mb-0")]
            course_decomp[key] = value

        return course_decomp

# gather PyTorch tutorials.
class PyTorchTutorialsParser:
    def __init__(self, tut_urls: Dict[str, Dict[str, str]]=SOURCE_TO_BASE_URLS["torch"]):
        self.tut_urls = tut_urls
        self.tut_pages = {}

    def download(self):
        for name, sub_blogs in tqdm(self.tut_urls.items()):
            self.tut_pages[name] = {}
            for sub_blog_name, url in tqdm(sub_blogs.items(), desc=name):
                simple_soup = simplify_soup(bs4.BeautifulSoup(
                    requests.get(url).text,
                    features="lxml",
                ), target="torch")

                simplified_soup = str(simple_soup)
                simplified_soup = re.sub("<div.*?>", "", simplified_soup)
                simplified_soup = re.sub("</div>", "", simplified_soup)
                simplified_soup = re.sub("<article.*?>", "", simplified_soup)
                simplified_soup = re.sub("</article>", "", simplified_soup)
                try: 
                    nb_json = extract_notebook_hierarchy_from_seq(
                        parse_soup_stream(simplified_soup)
                    )[0].serialize2()[""]
                    actual_content = None
                    found_keys = []
                    for ele in nb_json:
                        if isinstance(ele, dict):
                            key = process_text(list(ele.keys())[0].strip())
                            found_keys.append(key)
                            if key == sub_blog_name:
                                actual_content = list(ele.values())[0]
                                break
                            else: print(key, sub_blog_name)
                    assert actual_content is not None, f"{sub_blog_name} not found in {found_keys}"
                    self.tut_pages[name][sub_blog_name] = actual_content
                except IndexError:
                    print(f"ERROR[{name}][{sub_blog_name}]")
                    self.tut_pages[name][sub_blog_name] = simplified_soup

# gather NumPy tutorials.
class NumPyTutorialsParser:
    def __init__(self, blog_urls: Dict[str, Dict[str, str]]=SOURCE_TO_BASE_URLS["numpy"]):
        self.blog_urls = blog_urls
        self.blog_pages = {}

    def download(self):
        for name, sub_blogs in tqdm(self.blog_urls.items()):
            self.blog_pages[name] = {}
            for sub_blog_name, url in tqdm(sub_blogs.items(), desc=name):
                simple_soup = simplify_soup(bs4.BeautifulSoup(
                    requests.get(url).text,
                    features="lxml",
                ), target="numpy")

                simplified_soup = str(simple_soup)
                simplified_soup = re.sub("<div.*?>", "", simplified_soup)
                simplified_soup = re.sub("</div>", "", simplified_soup)
                simplified_soup = re.sub("<article.*?>", "", simplified_soup)
                simplified_soup = re.sub("</article>", "", simplified_soup)
                try: 
                    nb_json = extract_notebook_hierarchy_from_seq(
                        parse_soup_stream(simplified_soup)
                    )[0].serialize2()[""][0]
                    assert len(nb_json) == 1
                    value = list(nb_json.values())[0]
                    self.blog_pages[name][sub_blog_name] = value
                except IndexError:
                    print(f"ERROR[{name}][{sub_blog_name}]")
                    self.blog_pages[name][sub_blog_name] = simplified_soup

# gather matplotlib tutorials.
class MatPlotLibParser:
    """Gather matplotlib tutorial guides:"""
    def __init__(self, base_url: str=SOURCE_TO_BASE_URLS["matplotlib"]):
        self.base_url = base_url
        self.graph = {"Tutorials": []}
        self.base_soup = bs4.BeautifulSoup(
            requests.get(base_url).text,
            features="lxml",
        ) # return
        article = self.base_soup.html.body.select("article.bd-article")[0]
        tut_section = article.find("section", id="tutorials")
        for p in tut_section.select("p")[:2]:
            self.graph["Tutorials"].append((
                p.text, "markdown",
            ))
        self.graph["Tutorials"].append({})
        domain_dir = "https://matplotlib.org/stable/tutorials/"
        for section in tqdm(tut_section.select("section")):
            print(section.attrs["id"])
            h2 = section.select("h2")[0].text.strip("#")
            self.graph["Tutorials"][-1][h2] = []
            for p in section.select("p"):
                self.graph["Tutorials"][-1][h2].append((
                    p.text.strip(),
                    "markdown",
                ))
            self.graph["Tutorials"][-1][h2].append({})
            for div in section.select("div.sphx-glr-thumbcontainer"):
                key = div.select("div.sphx-glr-thumbnail-title")[0].text.strip("#")
                url = div.select("a.reference.internal")[0].attrs["href"]
                url = os.path.join(domain_dir, url)
                self.graph["Tutorials"][-1][h2][-1][key] = url 

    def download(self):
        for h2 in self.graph["Tutorials"][-1]:
            for key, url in tqdm(self.graph["Tutorials"][-1][h2][-1].items(), desc=h2):
                simple_soup = simplify_soup(
                    bs4.BeautifulSoup(
                        requests.get(url).text,
                        features="lxml",
                    ), target="matplotlib",
                )
                simplified_soup = str(simple_soup)
                simplified_soup = re.sub("<div.*?>", "", simplified_soup)
                simplified_soup = re.sub("</div>", "", simplified_soup)
                simplified_soup = re.sub("<article.*?>", "", simplified_soup)
                simplified_soup = re.sub("</article>", "", simplified_soup)
                simplified_soup = re.sub("<section.*?>", "", simplified_soup)
                simplified_soup = re.sub("</section>", "", simplified_soup)
                try:
                    nb_json = extract_notebook_hierarchy_from_seq(
                        parse_soup_stream(simplified_soup)
                    )[0].serialize2()[""][0]
                    assert len(nb_json) == 1 and isinstance(nb_json, dict)
                    value = list(nb_json.values())[0]
                    self.graph["Tutorials"][-1][h2][-1][key] = value
                except Exception as e:
                    print(e)
                    self.graph["Tutorials"][-1][h2][-1][key] = (simple_soup, simplified_soup)

# gather statsmodels: getting started, user guide and examples.
class StatsModelsParser:
    """Parser/scraper for statsmodels resources."""
    def __init__(self, base_urls: dict=SOURCE_TO_BASE_URLS["statsmodels"]):
        self.base_urls = base_urls
        self.user_guide_path = "./scrape_tutorials/statsmodels_user_guide.json"
        self.examples_path = "./scrape_tutorials/statsmodels_examples.json"
        self.graph = {}
        if os.path.exists(self.user_guide_path):
            self.user_guide_urls = json.load(open(self.user_guide_path))
        else:
            url = self.base_urls["User Guide"]
            self.user_guide_urls = self.scrape_user_guide_urls(url)
            with open(self.user_guide_path, "w") as f:
                json.dump(self.user_guide_urls, f, indent=4)
        if os.path.exists(self.examples_path):
            self.examples_urls = json.load(open(self.examples_path))
        else:
            url = self.base_urls["Examples"]
            self.examples_urls = self.scrape_examples_urls(url)
            with open(self.examples_path, "w") as f:
                json.dump(self.examples_urls, f, indent=4)
        self.graph.update(self.user_guide_urls)
        self.graph.update(self.examples_urls)
        self.graph["Getting Started"] = self.base_urls["Getting Started"]

    def download(self):
        self.graph["Getting Started"] = self.scrape_page(self.graph["Getting Started"])

    def scrape_page(self, url: str) -> dict:
        simple_soup = simplify_soup(bs4.BeautifulSoup(
            requests.get(url).text,
            features="lxml",
        ), target="statsmodels")
        simplified_soup = str(simple_soup)
        simplified_soup = re.sub("<div.*?>", "", simplified_soup)
        simplified_soup = re.sub("</div>", "", simplified_soup)
        simplified_soup = re.sub("<article.*?>", "", simplified_soup)
        simplified_soup = re.sub("</article>", "", simplified_soup)
        simplified_soup = re.sub("<section.*?>", "", simplified_soup)
        simplified_soup = re.sub("</section>", "", simplified_soup)

        nb_json = extract_notebook_hierarchy_from_seq(
                parse_soup_stream(simplified_soup)
            )[0].serialize2()[""]

        return nb_json
        # try:
        #     nb_json = extract_notebook_hierarchy_from_seq(
        #         parse_soup_stream(simplified_soup)
        #     )[0].serialize2()[""][0]
        #     assert len(nb_json) == 1 and isinstance(nb_json, dict)
        #     value = list(nb_json.values())[0]
        #     self.graph["Tutorials"][-1][h2][-1][key] = value
    def scrape_examples_urls(self, url: str):
        example_urls = {"Examples": []}
        soup = bs4.BeautifulSoup(
            requests.get(url).text,
            features="lxml",
        )
        domain_dir = "https://www.statsmodels.org/stable/"
        examples_section = soup.find("section", id="examples")
        for p in examples_section.select("p"):
            example_urls["Examples"].append((
                p.text,
                "markdown"
            ))
        example_urls["Examples"].append({})
        for section in examples_section.select("section"):
            h2 = section.select("h2")[0].text.replace("¶","")
            example_urls["Examples"][-1][h2] = {}
            for div in section.select("div.example.docutils.container"):
                p = div.select("p")[0]
                a = p.select('a.reference.external')[0]
                key = p.text.replace("¶","")
                example_urls["Examples"][-1][h2][key] = os.path.join(domain_dir, a.attrs["href"])

        return example_urls

    def scrape_user_guide_urls(self, url: str):
        user_guide_urls = {"User Guide": {}}
        soup = bs4.BeautifulSoup(
            requests.get(url).text,
            features="lxml",
        )
        domain_dir = "https://www.statsmodels.org/stable/"
        user_guide_section = soup.find("section", id="user-guide")
        for section in user_guide_section.select("section"):
            h2 = section.select("h2")[0].text.replace("¶","")
            user_guide_urls[h2] = {}
            for a in section.select("a.reference.internal"):
                key = a.text.replace("¶","")
                value = os.path.join(domain_dir, a.attrs["href"])
                user_guide_urls[h2][key] = value

        return user_guide_urls

# gather pandas tutorials from Tom's Blog.
class PandasTomsBlogParser:
    """Parser/scraper for Tom's blog tutorials for Pandas"""
    def __init__(self, blog_urls: Dict[str, str]=SOURCE_TO_BASE_URLS["pandas"]):
        self.blog_urls = blog_urls
        self.blog_pages = {}
        
    def download(self):
        for name, url in tqdm(self.blog_urls.items()):
            self.blog_pages[name] = {}
            simple_soup = simplify_soup(bs4.BeautifulSoup(
                requests.get(url).text,
                features="lxml",
            ), target="pandas")

            simplified_soup = str(simple_soup)
            simplified_soup = re.sub("<div.*?>", "", simplified_soup)
            simplified_soup = re.sub("</div>", "", simplified_soup)
            nb_json = extract_notebook_hierarchy_from_seq(
                parse_soup_stream(simplified_soup)
            )[0].serialize2()[""]
            self.blog_pages[name] = nb_json

# gather seaborn tutorials: https://seaborn.pydata.org/tutorial
class SeabornParser:
    """Parser/scraper for seaborn tutorials."""
    def __init__(self, base_url: str=SOURCE_TO_BASE_URLS["seaborn"]):
        self.base_url = base_url
        self.base_page = requests.get(base_url)
        self.base_soup = bs4.BeautifulSoup(
            self.base_page.text,
            features="lxml",
        )
        sections = [section for section in self.base_soup.find_all("section")]
        self.topic_urls = {}
        for section in sections:
            base_urls = {}
            name = section.find("h1")
            if name is None: name = section.find("h2")
            name = name.text.strip("#")
            for url in section.find_all("a", {"class": "reference internal"}): 
                base_host = "https://seaborn.pydata.org/"
                base_url = os.path.join(base_host, url.attrs['href'])
                base_url = urldefrag(base_url).url
                if base_url not in base_urls:
                    base_urls[base_url] = url.text
            self.topic_urls[name] = base_urls
        self.topic_pages = defaultdict(lambda:{})

    def download(self):
        for name, urls in self.topic_urls.items():
            for i, url in tqdm(enumerate(urls), total=len(urls)):
                simple_soup =  simplify_soup(bs4.BeautifulSoup(
                    requests.get(url).text,
                    features="lxml",
                ))
                simplified_soup = str(simple_soup).replace("<pre>", '''{"cell_type": "code", "code":\'\'\'''').replace("</pre>", '\'\'\'},')
                # for j in range(1, 12):
                    # simplified_soup = simplified_soup.replace(f"<h{j}>",  '{"cell_type": "markdown", "nl_original":\'\'\''+'#'*j+' ').replace(f"</h{j}>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h1>",  '{"cell_type": "markdown", "nl_original":\'\'\'# ').replace("</h1>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h2>",  '{"cell_type": "markdown", "nl_original":\'\'\'## ').replace("</h2>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h3>",  '{"cell_type": "markdown", "nl_original":\'\'\'### ').replace("</h3>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h4>",  '{"cell_type": "markdown", "nl_original":\'\'\'#### ').replace("</h4>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h5>",  '{"cell_type": "markdown", "nl_original":\'\'\'##### ').replace("</h5>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h6>",  '{"cell_type": "markdown", "nl_original":\'\'\'###### ').replace("</h6>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h7>",  '{"cell_type": "markdown", "nl_original":\'\'\'####### ').replace("</h7>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<h8>",  '{"cell_type": "markdown", "nl_original":\'\'\'######## ').replace("</h8>", '\'\'\'},')
                simplified_soup = simplified_soup.replace("<p>",  '{"cell_type": "markdown", "nl_original":\'\'\'').replace("</p>", '\'\'\'},')
                simplified_soup = re.sub("<section.*?>", "", simplified_soup)
                simplified_soup = simplified_soup.replace("</section>", "")
                # remove svgs, images, tables and style tags.
                simplified_soup = re.sub("<svg.*?>.*?</svg>", "", simplified_soup, flags=re.MULTILINE)
                # simplified_soup = re.sub("<style.*?>.*?</style>", "", simplified_soup, flags=re.MULTILINE)
                # simplified_soup = re.sub("<table.*?>.*?</table>", "", simplified_soup, flags=re.MULTILINE)
                simplified_soup = re.sub("<img.*?/>", "", simplified_soup)
                # html_to_json_dict = html_to_json.convert(str(simplified_soup))
                # simplified_json = simplify_html_to_json(html_to_json_dict)
                # simplified_json = collapse_list_of_strings(simplified_json)
                # self.topic_pages[name].append(simplified_json)
                
                try: 
                    nb_json = extract_notebook_hierarchy_from_seq(
                        ast.literal_eval("["+simplified_soup+"]")
                    )[0].serialize2()[""][0]
                    # print(nb_json)
                    assert len(nb_json.keys()) == 1
                    key = list(nb_json.keys())[0]
                    value = list(nb_json.values())[0] 
                    self.topic_pages[name][key] = value
                except SyntaxError:
                    # simplified_soup = re.sub("<a.*?>.*?</a>", "", simplified_soup)
                    # nb_json = extract_notebook_hierarchy_from_seq(
                    #     ast.literal_eval("["+simplified_soup+"]")
                    # )[0].serialize2()[""]
                    # assert len(nb_json.keys()) == 1
                    # key = list(nb_json.keys())
                    # value = list(nb_json.values()) 
                    # self.topic_pages[name][key] = value

                    
                    # try: self.topic_pages[name].append(extract_notebook_hierarchy_from_seq(
                    #     ast.literal_eval("["+simplified_soup+"]")
                    # )[0].serialize2()[""])
                    # except SyntaxError: 
                    print("ERROR:", name, i, url)
                        
                # self.topic_pages[name].append(
                #     simplify_bs2_json(
                #         BS2Json(
                #             requests.get(url).text
                #         ).convert()["html"]
                #     )
                # )
        self.topic_pages = dict(self.topic_pages)

def scrape_seaborn():
    """scrape seaborn data"""
    seaborn_parser = SeabornParser()
    seaborn_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    final_KG_json = {}
    for topic, page in seaborn_parser.topic_pages.items():
        final_KG_json[topic] = page
    with open("./scrape_tutorials/KGs/seaborn.json", "w") as f:
        json.dump(final_KG_json, f, indent=4)

def scrape_toms_blog_pandas():
    """scrape Pandas blogs from Tom's blog"""
    pandas_parser = PandasTomsBlogParser()
    pandas_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    final_KG_json = {}
    for topic, page in pandas_parser.blog_pages.items():
        final_KG_json[topic] = page
    with open("./scrape_tutorials/KGs/pandas_toms_blog.json", "w") as f:
        json.dump(final_KG_json, f, indent=4)

def scrape_numpy():
    """scrape NumPy tutorials"""
    numpy_parser = NumPyTutorialsParser()
    numpy_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    final_KG_json = {}
    for topic, page in numpy_parser.blog_pages.items():
        final_KG_json[topic] = page
    with open("./scrape_tutorials/KGs/numpy.json", "w") as f:
        json.dump(final_KG_json, f, indent=4)

def scrape_torch():
    """scrape PyTorch tutorials"""
    torch_parser = PyTorchTutorialsParser()
    torch_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    final_KG_json = {}
    for topic, page in torch_parser.tut_pages.items():
        final_KG_json[topic] = page
    with open("./scrape_tutorials/KGs/torch.json", "w") as f:
        json.dump(final_KG_json, f, indent=4)

def scrape_sklearn():
    """scrape SciKit learn examples and tutorials"""
    sklearn_parser = SciKitLearnParser()
    sklearn_parser.download_examples()
    sklearn_parser.download_tutorials()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    with open("./scrape_tutorials/KGs/sklearn.json", "w") as f:
        json.dump(sklearn_parser.graph, f, indent=4)

def scrape_matplotlib():
    """scrape matplotlib tutorials"""
    mplib_parser = MatPlotLibParser()
    mplib_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    with open("./scrape_tutorials/KGs/matplotlib.json", "w") as f:
        json.dump(mplib_parser.graph, f, indent=4)

def scrape_statsmodels():
    """scrape statsmodels examples, user guide and getting started guide"""
    sm_parser = MatPlotLibParser()
    sm_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    with open("./scrape_tutorials/KGs/statsmodels.json", "w") as f:
        json.dump(sm_parser.graph, f, indent=4)
    
def scrape_scipy():
    """scrape SciPy tutorials"""
    scipy_parser = SciPyParser()
    scipy_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    final_KG_json = {}
    for topic, page in scipy_parser.base_pages.items():
        final_KG_json[topic] = page
    with open("./scrape_tutorials/KGs/scipy.json", "w") as f:
        json.dump(final_KG_json, f, indent=4)

# main.
if __name__ == "__main__":
    # scrape_seaborn()        
    # scrape_toms_blog_pandas()
    # scrape_torch()
    # scrape_scipy()
    # scrape_sklearn()
    # scrape_matplotlib()
    scrape_statsmodels()
    # pass