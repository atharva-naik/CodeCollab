# collect Code Review data from Pull Requests.

import os
import json
from typing import *
from tqdm import tqdm
# Authentication is defined via github.Auth
from github import Auth
from github import Github

# scraper class.
class GithubPullRequestCommentsScraper:
    def __init__(self, github_object):
        self.data = []
        self.g = github_object

    def run(self, repo_names: List[str], file_name: str, 
            reset_file: bool=False, filt_no_comments: bool=True):
        if reset_file: open(file_name, "w")
        for repo_name in repo_names:
            repo = self.g.get_repo(repo_name)
            # main_pulls = repo.get_pulls(state='closed', base='main')
            # master_pulls = repo.get_pulls(state='closed', base='master')
            all_closed_pulls = repo.get_pulls(state='closed')
            for pr in tqdm(all_closed_pulls, desc=repo_name):
                rec = {}
                rec["pr_id"] = pr.id
                rec["body"] = pr.body
                rec["user"] = pr.user.login
                rec["pr_title"] = pr.title
                rec["repo_name"] = repo_name
                rec["comments"] = [self.extract_comment_json(comment) for comment in pr.get_review_comments()]
                if filt_no_comments and len(rec["comments"]) == 0: continue
                self.data.append(rec)
                with open(file_name, "a") as f:
                    f.write(json.dumps(rec)+"\n")
            # for pr in master_pulls:
            #     rec = {}
            #     rec["repo_name"] = repo_name
            #     rec["user"] = pr.user.login
            #     rec["pr_title"] = pr.title
            #     rec["comments"] = [self.extract_comment_json(comment) for comment in pr.get_comments()]
            #     self.data.append(rec)

    def extract_comment_json(self, comment):
        return {
            "user": comment.user.login if comment.user is not None else None,
            "diff": comment.diff_hunk,
            "body": comment.body,
            "id": comment.id,
            "url": comment.url,
        }

# main
if __name__ == "__main__":
    # using an access token
    creds = json.load(open("datautils/scraping/gh_access_token.json"))
    auth = Auth.Token(creds["access_token"])

    # Public Web Github
    g = Github(auth=auth)

    scraper = GithubPullRequestCommentsScraper(g)
    scraper.run(repo_names=["PyGithub/PyGithub"], file_name="./data/Comment_Generation/test_pr_scraper.jsonl", reset_file=True)