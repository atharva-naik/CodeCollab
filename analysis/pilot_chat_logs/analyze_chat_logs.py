# parse chat logs into a list of dialogs.
import json
import numpy as np
import pandas as pd
from typing import *
from collections import defaultdict

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

def compute_stats(dialog: List[Tuple[str, str]]):
    # turn_changes: 
    # average_uttr_len:
    uttr_lens = []
    turn_changes = 0
    prev_speaker = None
    per_speaker_uttr = defaultdict(lambda:0)
    for speaker, uttr in dialog:
        if prev_speaker != speaker:
            turn_changes += 1
        uttr_lens.append(len(uttr.split()))
        per_speaker_uttr[speaker] += 1
        prev_speaker = speaker
    print(dict(per_speaker_uttr))
    print(f"dialog len: {len(dialog)}")
    print(f"turn changes: {turn_changes}")
    print(f"mean utterance length: {np.mean(uttr_lens):.2f} ± {(np.var(uttr_lens)**0.5):.2f}")
    print(f"median utterance length: {np.median(uttr_lens)}")
    print(f"max utterance length: {np.max(uttr_lens):.2f}")
    print(f"min utterance length: {np.min(uttr_lens):.2f}")

# main 
if __name__ == "__main__":
    # Yiqing, Kelly, Luke
    dialog = parse_chat_log("./analysis/pilot_chat_logs/team_1_chat_log.txt")
    # print(dialog)
    compute_stats(dialog)
    df = []
    for i, (speaker, uttr) in enumerate(dialog):
        df.append({"id": i+1, "speaker": speaker, "utterance": uttr, "role": "", "comments": ""})
    df = pd.DataFrame(df)
    df.to_csv("./analysis/pilot_chat_logs/team_1_chat_log.xslx", index=False)
    # Atharva, Sireesh, James
    dialog = parse_chat_log("./analysis/pilot_chat_logs/team_2_chat_log.txt")
    # print(dialog)
    compute_stats(dialog)
    df = []
    for i, (speaker, uttr) in enumerate(dialog):
        df.append({"id": i+1, "speaker": speaker, "utterance": uttr, "role": "", "comments": ""})
    df = pd.DataFrame(df)
    df.to_csv("./analysis/pilot_chat_logs/team_2_chat_log.xslx", index=False)