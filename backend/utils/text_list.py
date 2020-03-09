import json
import re
import csv


def load_text(args):
    text_dict = {}
    # un_text = open("un_text", "r", encoding="utf-8")
    # un_text = open("call_2k/text/text", "r", encoding="utf-8")
    un_text = open(args.txtaddr, "r", encoding="utf-8")

    while True:
        line = un_text.readline()
        if not line:
            break
        line = line.strip()
        id = line.split(" ")[0]
        text = line[len(id) + 1 :]
        text_dict[id] = text
   # json.dump(text_dict, open("call_2k/text/text.json", "w", encoding="utf-8"))
    json.dump(text_dict, open(args.txjaddr, "w", encoding="utf-8"))
    return text_dict


def word2phone(args):
   # w2p_file = open("lexicon.txt", "r", encoding="utf-8")
    w2p_file = open(args.lexiconaddr, "r", encoding="utf-8")
    w2p_dict = {}
   # phone_file = open("phones.txt", "r", encoding="utf-8")
    phone_file = open(args.phonesaddr, "r", encoding="utf-8")
    phone_dict = {}
    reader = csv.reader(phone_file, delimiter=" ")
    for line in reader:
        phone_dict[line[0]] = int(line[1].strip())

    while True:
        line = w2p_file.readline()
        if not line:
            break
        line = line.strip()
        line = re.sub("  ", "\t", line)
        line = line.split("\t")
        if len(line) < 2:
            continue
        else:
            word = line[0]
            phones = line[1].split(" ")
            w2p_dict[word] = list(map(lambda x: phone_dict[x], phones))

    return w2p_dict


def text2phone(text, w2p):
    t2p = {}
    for key in text.keys():
        phone_list = [1]
        words = text[key].split(" ")
        try:
            for word in words:
                phone_list.extend(w2p[word])
                phone_list.append(1)
        except:
            print(words)
            print(text[key])
            print(key)
        t2p[key] = phone_list

    return t2p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('txjaddr', type=str, help='the addr of json of text')
    parser.add_argument('txtaddr', type=str, help='the addr of text')
    parser.add_argument('lexiconaddr', type=str, help='the addr of txt of lexicon')
    parser.add_argument('phonesaddr', type=str, help='the addr of txt of phones')
    parser.add_argument('outaddr', type=str, help='the addr of json of outputs')
    args = parser.parse_args()

    text = load_text(args)
    w2p = word2phone(args)
    t2p = text2phone(text, w2p)
   # json.dump(t2p, open("call_2k/text/text2phone.json", "w", encoding="utf-8"))
    json.dump(t2p, open(args.outaddr, "w", encoding="utf-8"))
