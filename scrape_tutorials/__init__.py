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
    },
    "numpy": {
        "NumPy Features": {
            "Linear algebra on n-dimensional arrays": "https://numpy.org/numpy-tutorials/content/tutorial-svd.html",
            "Saving and sharing your NumPy arrays": "https://numpy.org/numpy-tutorials/content/save-load-arrays.html",
            "Masked Arrays": "https://numpy.org/numpy-tutorials/content/tutorial-ma.html",
        },
        "NumPy Applications": {
            "Determining Moore’s Law with real data in NumPy": "https://numpy.org/numpy-tutorials/content/mooreslaw-tutorial.html",
            "Deep learning on MNIST": "https://numpy.org/numpy-tutorials/content/tutorial-deep-learning-on-mnist.html",
            "X-ray image processing": "https://numpy.org/numpy-tutorials/content/tutorial-x-ray-image-processing.html",
            "Determining Static Equilibrium in NumPy": "https://numpy.org/numpy-tutorials/content/tutorial-static_equilibrium.html",
            "Plotting Fractals": "https://numpy.org/numpy-tutorials/content/tutorial-plotting-fractals.html",
            "Analyzing the impact of the lockdown on air quality in Delhi, India": "https://numpy.org/numpy-tutorials/content/tutorial-air-quality-analysis.html",
        },
        "Articles": {
            "Deep reinforcement learning with Pong from pixels": "https://numpy.org/numpy-tutorials/content/tutorial-deep-reinforcement-learning-with-pong-from-pixels.html",
            "Sentiment Analysis on notable speeches of the last decade": "https://numpy.org/numpy-tutorials/content/tutorial-nlp-from-scratch.html",
        }
    }
}