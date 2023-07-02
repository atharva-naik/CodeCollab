import os
import json
import requests

REL_NODES = {
    "Q11660": "", # artificial intelligence
    "Q2539": "", # machine learning
    "Q2374463": "", # data science
    "Q12483": "", # statistics
    "Q35308049": "", # statistical data
    "Q208042": "", # regression analysis
    "Q12718609": "", # statistical method
    "Q1988917": "", # data analysis
    "Q485396": "", # analytics 
    "Q1149776": "", # data management
    "Q11661": "", # information technology
    "Q77293133": "", # data analyst
    "Q42848": "", # data
    "Q15088675": "", # data curation
    "Q190087": "", # data type
    "Q5227257": "", # data classification (data management)
    "Q494823": "", # data format
    "Q112598603": "", # data professional
    "Q188889": "", # code
    "Q1417149": "", # rule-based system
    "Q59154708": "", # data export
    "Q1783551": "", # data conversion
    "Q6661985": "", # data processing
    "Q750843": "", # information processing
    "Q107491038": "", # data processor
    "Q8366": "", # algorithm
    "Q5157286": "", # computational complexity
    "Q1296251": "", # algorithmic efficiency
    "Q59154760": "", # data import
    "Q235557": "", # file format
    "Q65757353": "", # transformation
    "Q7595718": "", # algorithmic stability
    "Q1412694": "", # knowledge-based system
    "Q217602": "", # analysis
}
PROPS = [
    "P361", # part of
    # "P910", # topic's main category
    "P279", # subclass of
    # "P1424", # topic's main template 
    # "P1482", # Stack Exchange Tag
    "P527", # has part(s)
    "P1889", # different from
    # "P373", # Commons Category.
    "P3095", # practiced by
    "P31", # instance of
    "P1552", # has quality
    "P2184", # history of topic
    # "P6541", # stack exchange site url
    "P366", # has use
    "P737", # influenced by
    "P797", # significant event
    "P3712", # has goal
]
BLOCK_LIST = [
    "Q11862829", # significant event
    "Q268592", # industry 
    "Q120208", # emerging technology
    "Q112057532", # type of technology
    "Q14623823", # artificiality 
]
# main
if __name__ == "__main__":
    url = "https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&entity={}"
    queue = [qid for qid in REL_NODES]
    for qid in queue:
        qjson = json.loads(requests.get(url.format(qid)).text)
        