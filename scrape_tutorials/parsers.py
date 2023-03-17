#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# parsers for various libraries/modules.
import os
import bs4
import bs2json
import requests
from typing import *
from tqdm import tqdm
from bs2json import BS2Json
from urllib.parse import urldefrag
from collections import defaultdict
from scrape_tutorials import SOURCE_TO_BASE_URLS

def simplify_soup(soup):
    for span in soup.select('span'):
        span.unwrap()
    for span in soup.select('code'):
        span.unwrap()
    for span in soup.select('div'):
        span.unwrap()
    for span in soup.select("p"):
        span.unwrap()

    article = soup.html.body.find("article", {"class": "bd-article"})
    for ele in article.select("article"): ele.unwrap()

    return article

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

class SeabornParser:
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
            base_urls = set()
            name = section.find("h1")
            if name is None: name = section.find("h2")
            name = name.text.strip("#")
            for url in section.find_all("a", {"class": "reference internal"}): 
                base_host = "https://seaborn.pydata.org/"
                base_url = os.path.join(base_host, url.attrs['href'])
                base_url = urldefrag(base_url).url
                base_urls.add(base_url)
            self.topic_urls[name] = sorted(list(base_urls))
        self.topic_pages = defaultdict(lambda:[])

    def download(self):
        for name, urls in self.topic_urls.items():
            for url in tqdm(urls):
                self.topic_pages[name].append(
                    simplify_soup(bs4.BeautifulSoup(
                        requests.get(url).text,
                        features="lxml",
                    ))
                )
                # self.topic_pages[name].append(
                #     simplify_bs2_json(
                #         BS2Json(
                #             requests.get(url).text
                #         ).convert()["html"]
                #     )
                # )
        self.topic_pages = dict(self.topic_pages)

# main.
if __name__ == "__main__":
    seaborn_parser = SeabornParser()
    seaborn_parser