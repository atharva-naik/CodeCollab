import csv
import json
import sys
import os

for row in csv.DictReader(open(sys.argv[1])):
    ext = json.loads(row["extensions"])
    codeUrl = ext["codeUrl"]
    print(row["module_slug"], row["task_slug"], row["score_obtained"], codeUrl) 
    os.system("wget " + codeUrl)
    quit()