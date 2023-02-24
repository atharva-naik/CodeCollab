# Manual analysis:
Manually categorizing error types:

## Step mismatch:
61-62 mismatch
61 true = 62 pred

```{"id": 2, "true": "If everything went well you should get a validation/test accuracy very close to 0.8.\n\n### 6.3 Random Forests\n\nAn often used technique in machine learning are random forests. Basically they are [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning), or in programmers terms, if-then-else structures, like the one shown below.\n\n<img src=\"images/tree.png\" width=70%>\n\nDecision trees are know to over-fit a lot because they just learn the train set by heart and store it. Random forests on the other hand combine multiple different (randomly initialized) decision trees that all over-fit in their own way. But by combining their output using a voting mechanism, they tend to cancel out eachothers mistakes. This approach is called an [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) and can be used for any combination of machine learning techniques. A schematical representation of how such a random forest works is shown below.\n\n<img src=\"images/random_forest.jpg\">\n\nNow let's try to use a random forest to solve the double spiral problem. (see [`sklearn.ensemble.RandomForestClassifier()`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))", "pred": "As you can see a non-linear classifier returns a non-linear decision boundary.\n\n### 6.3 Non-linear classification\n\nNow let's do this better with a non-linear classifier using polynomials. Play with the degree of the polynomial expansion and look for the effect of the `RidgeCV()` and `LassoCV()` models. What gives you the best results?"}```

```{"id": 2, "true": "As you can see they are quite powerful right out of the box without any parameter tuning. But we can get the results even beter with some fine tuning.\n\nTry changing the `min_samples_leaf` parameter for values between 0 and 0.5.", "pred": "If everything went well you should get a validation/test accuracy very close to 0.8.\n\n### 6.4 Gradient Boosting\n\nGradient boosting is an ensemble learning technique that combines multiple decision trees into a single decision tree. Random forests are often used for this purpose.\n\n<img src=\"images/gradient_boosting.jpg\">\n\nThe idea behind gradient boosting is to combine multiple decision trees into a single decision tree that learns the best combination of features. This is called a [boosting machine](https://en.wikipedia.org/wiki/Boosting_machine).\n\n### 6.5 Support Vector Machines\n\nSupport vector machines are a type of [decision tree](https://en.wikipedia.org/wiki/Decision"}```

observation appears earlier/later
point numbers mismatch 5A vs 5B (something like that)
 
## Attribute mismatch:
```{"id": 0, "true": "**5.B** Using your function, print the summary statistics for '2015_margin'", "pred": "**5.B** Using your function, print the summary statistics for 'sales_price'"}```
```{"id": 0, "true": "**5.C** Using your function, print the summary statistics for '2015_q1_sales'", "pred": "**5.C** Using your function, print the summary statistics for '2015_margin_per_sqft'"}```
```{"id": 0, "true": "**5.D** Using your function, print the summary statistics for '2016_q1_sales'", "pred": "**5.D** Using your function, print the summary statistics for '2015_q3_sales'"}```

## Hierarchy mismatch:
Number of hashes are different

## Incorrect numbering scheme/organization scheme:
```{"id": 3, "true": "### 5a) Select the data\n\nPull out the data we are going to use:\n- Save the height data for all 'COGSCI' majors to a variable called 'h_co'\n- Save the height data for all 'COMPSCI' majors to a variable called 'h_cs'", "pred": "## Part 5.1: Data Cleaning"}```

## Question type mismatch:

```{"id": 43, "true": "## Question 5\nWhat might be a problem with having some POIs missing from our dataset? (Quiz: Problems with Incomplete Data)\n\nThis is more of a \"free response\" thought question -- we don't really expect you to answer this using code.", "pred": "## Question 5\nHow many POIs are in the dataset that have email addresses?\n\nHint: Use the **poi_have_emails** list to see which POIs are in the dataset."}```

```{"id": 43, "true": "## Question 6\nWhat is the total value of the stock belonging to James Prentice? (Query The Dataset 1)", "pred": "## Question 6\nHow many POIs are missing from our dataset? (Quiz: How Many POIs Are Missing?)\n\n**Hint:** Open up the `poi_missing_emails.txt` file to see the file format:\n  - the first line is a link to a USA Today article\n  - the second line is blank\n  - subsequent lines have the format: `(\u2022) Lastname, Firstname`\n      - the dot `\u2022` is either \"y\" (for yes) or \"n\" (for no), describing if the emails for that POI are available"}```

## Repitive generations:

```{"id": 44, "true": "Now we have to standardize the data!", "pred": "df.isnull.any(axis=1) is now a boolean mask that tells you which rows have any empty or null values. \nThe boolean mask can be used in a boolean index to select rows or columns. \nThe boolean mask can be used in a boolean index to select rows or columns. \nThe boolean mask can be used in a boolean index to select rows or columns. \nThe boolean mask can be used in a boolean index to select rows or columns. \nThe boolean mask can be used in a boolean index to select rows or columns. \nThe boolean mask can be used in a boolean index to select rows or columns. \nThe boolean mask can be used in a boolean index to select rows or column"}```
```"Understand why you think this might be a good idea. \nUnderstand why you think this might be a bad idea. \nUnderstand why you think this might be a good idea. \nUnderstand why you think this might be a bad idea. \nUnderstand why you think this might be a good idea. \nUnderstand why you think this might be a bad idea. \nUnderstand why you think this might be a good idea. \nUnderstand why you think this might be a bad idea. \nUnderstand why you think this might be a good idea. \nUnder"```

# Automated analysis:
Trying to understand:
**Style Analysis:**
1. Hierarchy
2. Pointing/formatting scheme
**Content Analysis:**
1. Verbs match.
2. Keyphrases match.
3. Entity slot matches (need some more work)