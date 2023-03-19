#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# code for gathering, processing and indexing tutorials
# encode task decompositions

import json

SOURCE_TO_BASE_URLS = {
    "seaborn": "https://seaborn.pydata.org/tutorial",
    "pandas": {
        "Modern Pandas": "https://tomaugspurger.github.io/posts/modern-1-intro/",
        "Method Chaining": "https://tomaugspurger.github.io/posts/method-chaining/",
        "Indexes": "https://tomaugspurger.github.io/posts/modern-3-indexes/",
        "Fast Pandas": "https://tomaugspurger.github.io/posts/modern-4-performance/",
        "Tidy Data": "https://tomaugspurger.github.io/posts/modern-5-tidy/",
        "Visualization": "https://tomaugspurger.github.io/posts/modern-6-visualization/",
        "Time Series": "https://tomaugspurger.github.io/posts/modern-7-timeseries/",
        "Scaling": "https://tomaugspurger.github.io/posts/modern-8-scaling/", 
    }
}