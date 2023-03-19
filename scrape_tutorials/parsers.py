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
    for hr in soup.select("hr"): hr.unwrap()
    for span in soup.select('span'): span.unwrap()
    for code in soup.select('code'): code.unwrap()
    for div in soup.select('div'): div.unwrap()
    for li in soup.select('li'): li.unwrap()
    for ol in soup.select('ol'): ol.unwrap()
    for ul in soup.select('ul'): ul.unwrap()
    for table in soup.select('table'): table.extract()
    for style in soup.select('style'): style.extract()
    for a in soup.select('a'): a.extract()
    for pre in soup.select("pre"):
        del pre.attrs
    for p in soup.select("p"): 
        del p.attrs
    for i in range(1, 12):
        for hi in soup.select(f"h{i}"):
            del hi.attrs
    # for section in soup.select("section"): section.unwrap()

    if target == "seaborn": 
        article = soup.html.body.find("article", {"class": "bd-article"})
        for content in article.contents:
            if content != "\n":
                final = content
                break
    elif target == "pandas": final = soup

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

def simplify_html_to_json(html_to_json_dict: dict):
    simple_json = {}
    # print("simplifying JSON")
    if isinstance(html_to_json_dict, dict):
        if len(html_to_json_dict) == 1 and isinstance(list(html_to_json_dict.values())[0], str):
            return list(html_to_json_dict.values())[0]
        for key, value in html_to_json_dict.items():
            if key == "_value": key = "content"
            elif key == "h1": key = "markdown1"
            elif key == "h2": key = "markdown2"
            elif key == "h3": key = "markdown3"
            elif key == "_values": key = "contents"
            elif key == 'pre': key = "code"
            # print(key)
            if isinstance(value, (dict, list)):
                simple_json[key] = simplify_html_to_json(value)
            else: simple_json[key] = value
    elif isinstance(html_to_json_dict, list):
        simple_json = []
        for value in html_to_json_dict:
            simple_json.append(simplify_html_to_json(value))
        return simple_json
    else: return html_to_json_dict

    return simple_json

def simplify_bs2_json(html_json: str, filt_keys: List[str]=[
                      'script', 'style', 'role', 'br', 'class']):
    if isinstance(html_json, dict):
        if [k for k in html_json.keys() if k not in filt_keys] == ["text"]:
            if isinstance(html_json["text"], list):
                return " ".join(html_json["text"])    
            return html_json["text"]
        new_json = {}
        for key, value in html_json.items():
            if key in filt_keys: 
                # print(key)
                pass
            elif key == "attrs":
                for k,v in html_json["attrs"].items():
                    if k in filt_keys: continue
                    new_json[k] = simplify_bs2_json(v)
            elif isinstance(value, (dict, list)):
                new_json[key] = simplify_bs2_json(value, filt_keys)
            elif key == "p" and isinstance(value, list):
                new_json[key] = "\n".join(value)
            elif key == "text" and isinstance(value, list):
                new_json[key] = " ".join(value)
            else: new_json[key] = value
    elif isinstance(html_json, list):
        new_json = []
        for value in html_json:
            new_json.append(simplify_bs2_json(value, filt_keys))
    else:
        # print(type(html_json))
        new_json = html_json

    return new_json

# def unenclosed_dict_error_correction(data_stream: str):
#     bracket_stack = []
#     i = 0
#     new_stream = ""
#     while i < len(data_stream):
#         char = data_stream[i]
#         if char == "{":
#             bracket_stack.append("{")
#             new_stream += char
#         elif char == "}":
#             if len(bracket_stack) > 0:
#                 bracket_stack.pop()
#                 new_stream += char
#                 i += 1
#                 char = data_stream[i]
#                 if char == ",": new_stream += char
#                 else: new_stream += (","+char)
#             else: pass
#         i += 1

#     return new_stream

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
            # remove any remaining div tags.
            simplified_soup = re.sub("<div.*?>", "", simplified_soup)
            simplified_soup = re.sub("</div>", "", simplified_soup)
            # simplified_soup = re.sub("<style.*?>.*?</style>", "", simplified_soup, flags=re.MULTILINE)
            # simplified_soup = re.sub("<table.*?>.*?</table>", "", simplified_soup, flags=re.MULTILINE)
            simplified_soup = re.sub("<img.*?/>", "", simplified_soup)

            try: 
                nb_json = extract_notebook_hierarchy_from_seq(
                    ast.literal_eval("["+simplified_soup+"]")
                )[0].serialize2()[""][0]
                # print(nb_json)
                assert len(nb_json.keys()) == 1
                key = list(nb_json.keys())[0]
                value = list(nb_json.values())[0] 
                self.blog_pages[name][key] = value 
            except SyntaxError:
                print(f"ERROR: {name}({url})")
                self.blog_pages[name] = simplified_soup

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
    seaborn_parser = SeabornParser()
    seaborn_parser.download()
    os.makedirs("./scrape_tutorials/KGs", exist_ok=True)
    final_KG_json = {}
    for topic, page in seaborn_parser.topic_pages.items():
        final_KG_json[topic] = page
    with open("./scrape_tutorials/KGs/seaborn.json", "w") as f:
        json.dump(final_KG_json, f, indent=4)

# main.
if __name__ == "__main__":
    scrape_seaborn()        