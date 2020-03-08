import json
import csv


if __name__ == "__main__":
    data = json.load(open("call_2k/scores.json", "r", encoding="utf-8"))
    out = open("call_2k/score_analysis.csv", "w", encoding="utf-8", newline="")
    writer = csv.writer(out)
    header = [
        "id", "text", "words_number", "phones_number", "vowel", "consonant",
        "pronunciation", "integrity", "fluency"
    ]
    writer.writerow(header)
    for key in data.keys():
        out_list = [key, data[key]["text"], len(data[key]["words"])]
        phone_number = 0
        vowel = 0
        consonant = 0
        for word in data[key]["words"]:
            phone_number += len(word["phones"])
            for phone in word["phones"]:
                if len(phone["phone"]) > 2:
                    vowel += 1
                else:
                    consonant += 1
        out_list.extend([phone_number, vowel, consonant,
                         data[key]["pronunciation"], data[key]["integrity"], data[key]["fluency"]])
        writer.writerow(out_list)
