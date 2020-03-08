import json
import math


def pscore(prob_segment, phone_id):
    if len(prob_segment) == 0:
        return 0
    length =len(prob_segment)
    score_start = math.floor(length / 4)
    score_end = min(math.floor(3 * length / 4) + 1, length)
    score_prob = prob_segment[score_start:score_end]
    target_result = []
    for prob in score_prob:
        if prob[phone_id - 1] > 0.5:
            target_result.append(100)
        else:
            if prob[phone_id - 1] < 1e-8:
                target_result.append(0)
                continue
            target_result.append(math.log(max(prob)) / math.log(prob[phone_id - 1]) * 100)
    target_result = sorted(target_result)
    return round(sum(target_result[:min(1, len(target_result))]) / min(len(target_result), 1), 4)


def cal_fluency(result_dict, ar):
    if len(result_dict["words"]) == 1:
        return 100
    content = result_dict["end"] - result_dict["start"]
    silence_sum = 0
    for p in ar[result_dict["start"]:result_dict["end"]]:
        if p == 1:
            silence_sum += 1
    standard_size = 3 * len(result_dict["words"]) - 3
    silence_sum = max(silence_sum - standard_size, 0)
    fluency = max(1- silence_sum / (content - silence_sum), 0) * 100
    return fluency


if __name__ == "__main__":
    aligned_result = json.load(open("call_2k/aligned/aligned_result_0.json", "r", encoding="utf-8"))
    text = json.load(open("call_2k/text/text.json", "r", encoding="utf-8"))
    probs = json.load(open("call_2k/feature/feature_0.json", "r", encoding="utf-8"))
    phones = json.load(open("phones.json", "r", encoding="utf-8"))
    tphones = json.load(open("call_2k/text/text2phone.json", "r", encoding="utf-8"))
    reverse_phone = {}
    for key in phones.keys():
        reverse_phone[phones[key]] = key
    output = {}
    for key in aligned_result.keys():
        ar = aligned_result[key]
        prob = probs[key]
        if len(key) == 11:
            key = key[:7] + key[8:]
        template = tphones[key]
        txt = text[key]
        ps = tphones[key]
        

        # construt result_dict
        words = txt.split(" ")
        result_dict = {}
        result_dict["words"] = []
        result_dict["text"] = txt
        template_position = 0
        align_position = 0
        print(txt)
        print(ar)

        # skip the first silence
        while ar[align_position] == 1:
            align_position += 1
        template_position += 1

        for word in words:
            word_dict = {}
            word_dict["word"] = word
            word_dict["phones"] = []
            while template_position < len(template):
                # TODO: continued phones - same phone output
                if template[template_position] == template[template_position - 1]:
                    template_position += 1
                    continue
                # skip silence phone
                if template[template_position] == 1:
                    while align_position < len(ar) and ar[align_position] == 1:
                        align_position += 1
                    template_position += 1
                    break
                phone_dict = {}
                phone_dict["phone"] = str(reverse_phone[template[template_position]])
                phone_dict["start"] = align_position
                while align_position < len(ar) and ar[align_position] == template[template_position]:
                    align_position += 1
                phone_dict["end"] = align_position
                # print(ar)
                # print(key)
                # print(word)
                # print(phone_dict["phone"])
                # print(phone_dict["start"])
                # print(phone_dict["end"])
                # print(txt)
                print("-------------")
                phone_dict["score"] = pscore(prob[phone_dict["start"]:phone_dict["end"]], template[template_position])
                word_dict["phones"].append(phone_dict)
                template_position += 1
            vowels = []
            consonant = []
            print(word_dict["word"])
            print(word_dict["phones"])
            for phone in word_dict["phones"]:
                if len(phone["phone"]) > 2:
                    vowels.append(phone["score"])
                else:
                    consonant.append(phone["score"])
            # if len(consonant) == 0:
            #     word_dict["score"] = round(sum(vowels) / len(vowels), 4)
            # elif len(vowels) == 0:
            #     word_dict["score"] = round(sum(consonant) / len(consonant), 4)
            # else:
            #     word_dict["score"] = round(0.75 * sum(vowels) / len(vowels) + 0.25 * sum(consonant) / len(consonant), 4)
            word_dict["score"] = round((sum(vowels) + sum(consonant)) / (len(vowels) + len(consonant)), 4)
            word_dict["start"] = (word_dict["phones"])[0]["start"]
            word_dict["end"] = (word_dict["phones"])[-1]["end"]
            result_dict["words"].append(word_dict)
        word_score = []
        inter_word = 0
        phones_count = 0
        phones_sum = 0
        for word in result_dict["words"]:
            word_score.append(word["score"])
            phones_count += len(word["phones"])
            phones_sum += len(word["phones"]) * word["score"]
            if word["score"] > 25:
                inter_word += 1
        result_dict["integrity"] = round(inter_word / len(result_dict["words"]) * 100, 4)
        result_dict["start"] = (result_dict["words"])[0]["start"]
        result_dict["end"] = (result_dict["words"])[-1]["end"]
        # result_dict["pronunciation"] = round(sum(word_score) / len(word_score), 4)
        result_dict["pronunciation"] = round(phones_sum / phones_count, 4)
        result_dict["fluency"] = round(cal_fluency(result_dict, ar), 4)
        json.dump(result_dict, open("score/%s.json" % key, "w", encoding="utf-8"))
        output[key] = result_dict
    print(output)
    json.dump(output, open("call_2k/scores.json", "w", encoding="utf-8"))
