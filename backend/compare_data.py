import json
import csv


if __name__ == "__main__":
    hscore = open("human-rated.csv", "r", encoding="utf-8")
    hjson = {}
    reader = csv.reader(hscore)
    header = next(reader)
    for line in reader:
        if line[3] == '':
            continue
        hjson[line[0]] = [int(line[1]) * 10, int(line[2]) * 10, int(line[3]) * 10]

    cscore = json.load(open("scores.json", "r", encoding="utf-8"))

    with open("compare.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "hp", "hf", "hi", "cp", "cf", "ci"])
        for key in hjson.keys():
            if key not in cscore.keys():
                continue
            writer.writerow([key, hjson[key][2], hjson[key][1], hjson[key][0],
                             cscore[key]["pronunciation"], cscore[key]["fluency"], cscore[key]["integrity"]])