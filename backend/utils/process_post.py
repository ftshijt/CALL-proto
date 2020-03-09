import os
import json


def clip(element):
    element = float(element)
    # return element
    if element > 1e-10:
        return element
    else:
        return 0.0


def sum(i, index, data, args):
   # data_source = open("../../exp/chain_cleaned/tdnn_1d_sp_post_call_1k/phone_post_%d.txt"%i, "r", encoding="utf-8")
    data_source = open("%s/phone_post_%d.txt"%(args.ppaddr, i), "r", encoding="utf-8")
    temp_data = []
    key = ""
    end = False
    while True:
        line = data_source.readline()
        if not line:
            break

        line = line[:-1]
        if '[' in line:
            # call 10
            # librispeech 15
            line = line.split(' ')
            key = line[0]
            print(key)
            temp_data = []
            end = False
            continue
        elif ']' in line:
            line = line[:-1]
            end = True
        line = line.strip().split(" ")
        line = list(map(clip, line))
        temp_data.append(line)
        if end:
            data[key] = temp_data
            if len(data.keys()) > 10000:
               # json.dump(data, open("call_2k/feature/feature_%d.json"%index, "w", encoding="utf-8"))
                json.dump(data, open("%s/feature_%d.json"%(args.feataddr, index), "w", encoding="utf-8"))
                index += 1
                data = {}
                print("feature--%d--finished"%index)
    return data, index

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ppaddr', type=str, help='the addr of txt of phone_post')
    parser.add_argument('feataddr', type=str, help='the addr of json of feature')
    args = parser.parse_args()

    data = {}
    index = 0
    for i in range(1, 2):
        data, index = sum(i, index, data, args)
   # json.dump(data, open("call_2k/feature/feature_%d.json"%index, "w", encoding="utf-8"))
    json.dump(data, open("%s/feature_%d.json"%(args.feataddr, index), "w", encoding="utf-8"))
        
