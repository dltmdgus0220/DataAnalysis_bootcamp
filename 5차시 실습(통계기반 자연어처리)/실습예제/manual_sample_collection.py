import pandas as pd
from pathlib import Path
import os, re

FOLDER_PATH = Path(r"D:\multicompus\example\nlp\stat_nlp\naive_bayes_svm\1-1. 여성의류")

TARGET_PER_CLASS = 600
TARGET_CLASS = [-1, 0, 1]

bucket = { c:[] for c in TARGET_CLASS}
count_per_class = { c : 0 for c in TARGET_CLASS}

for path in FOLDER_PATH.glob("*.json"):
    print(f"\n읽을 파일명:{path.name}")

    if all(count_per_class[c] >= TARGET_PER_CLASS for c in TARGET_CLASS):
        print("데이터 수집 완료")
        break
    
    df = pd.read_json(path, encoding="utf-8")
    if("GeneralPolarity" not in df.columns) or ("RawText" not in df.columns):
        print("GeneralPolarity 또는 RawText 컬럼 없음")
        continue

    df = df.dropna(subset=["GeneralPolarity", "RawText"]).copy()
    df["GeneralPolarity"] = df["GeneralPolarity"].astype(int)

    for c in TARGET_CLASS:
        if count_per_class[c] >= TARGET_PER_CLASS:
            continue

        need = TARGET_PER_CLASS - count_per_class[c]
        cand = df[df["GeneralPolarity"]==c]
        if(len(cand)) == 0:
            continue

        take_n = min(need, len(cand))
        sample = cand.sample(n=take_n, random_state=42)

        bucket[c].append(sample)
        count_per_class[c] += take_n

        print(f"클래스 : {c} : {take_n}개 추가(현재 {count_per_class[c]}개)")
    

dfs = []

for c in TARGET_CLASS:
    if bucket[c]:
        cls_df = pd.concat(bucket[c], ignore_index=True)
        dfs.append(cls_df)

df_final = pd.concat(dfs, ignore_index=True)

print("\n최종 클래스 분포:")
print(df_final["GeneralPolarity"].value_counts())
print("총 샘플 수 : ", len(df_final))

current_file_path = Path(__file__).resolve()
current_path = current_file_path.parent

# current_path = os.getcwd()

OUT_PATH = current_path.name + "\woman_wear_sample_600.json"
df_final.to_json(OUT_PATH, orient="records", force_ascii=False, indent=4)
print("저장완료: ", OUT_PATH)











