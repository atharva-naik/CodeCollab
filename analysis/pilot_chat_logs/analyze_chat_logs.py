# parse chat logs into a list of dialogs.
import json
import pandas as pd
from typing import *

def parse_chat_log(path: str) -> List[Tuple[str, str]]:
    dialogs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "says:" not in line: continue
            chunks = line.split("says:")
            assert len(chunks) == 2, f"{chunks} for {line}"
            speaker = chunks[0].strip()
            utterance = chunks[1].strip()
            if speaker == "me":
                speaker = "Armineh"
            dialogs.append((speaker, utterance))

    return dialogs

# main 
if __name__ == "__main__":
    # Yiqing, Kelly, Luke
    dialog = parse_chat_log("./analysis/pilot_chat_logs/team_1_chat_log.txt")
    print(dialog)
    df = []
    for i, (speaker, uttr) in enumerate(dialog):
        df.append({"id": i+1, "speaker": speaker, "utterance": uttr, "role": "", "comments": ""})
    df = pd.DataFrame(df)
    df.to_csv("./analysis/pilot_chat_logs/team_1_chat_log.xslx", index=False)
    # Atharva, Sireesh, James
    dialog = parse_chat_log("./analysis/pilot_chat_logs/team_2_chat_log.txt")
    print(dialog)
    df = []
    for i, (speaker, uttr) in enumerate(dialog):
        df.append({"id": i+1, "speaker": speaker, "utterance": uttr, "role": "", "comments": ""})
    df = pd.DataFrame(df)
    df.to_csv("./analysis/pilot_chat_logs/team_2_chat_log.xslx", index=False)