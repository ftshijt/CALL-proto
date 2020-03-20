import json
import re
import csv
import numpy as np
import math
import argparse


class Score():
    def __init__(self, args):
        self.phone_dict, self.w2p = self.word2phone(args)
       # self.phone_dict = self.load_phone(args)
        self.sr = args.sr
        # ...

    def cleantext(self, text):
        punctuation = '!,;:?"\''
        text = re.sub(r'[{}]+'.format(punctuation),'',text)
        return text.strip().upper()

    def word2phone(self, args):
        # w2p_file = open("lexicon.txt", "r", encoding="utf-8")
        w2p_file = open(args.lexiconaddr, "r", encoding="latin-1")
        w2p_dict = {}
       # phone_file = open("phones.txt", "r", encoding="utf-8")
        phone_file = open(args.phonesaddr, "r", encoding="utf-8")
        phone_dict = {}
        reader = csv.reader(phone_file, delimiter=" ")
        id_num = 0
        for line in reader:
            phone_dict[line[0]] = int(id_num)
            id_num += 1

        while True:
            line = w2p_file.readline()
            if not line:
                break
            line = line.strip()
            line = re.sub("  ", "\t", line)
            line = line.split("\t")
            if len(line) < 2:
                continue
            if ';;;' in line[0]:
                continue
            else:
                word = line[0]
                phones = line[1].split(" ")
                w2p_dict[word] = list(map(lambda x: phone_dict[x], phones))

        return phone_dict, w2p_dict


    def text2phone(self, text):
        #TODO
        text = self.cleantext(text)
        phone_list = [1]     ###### why？？？
        words = text.split(" ")
        try:
            for word in words:
                phone_list.extend(self.w2p[word])
                phone_list.append(1)
        except:
            print(words)
            print(text)
        t2p = phone_list

        return t2p

    def kaldi_infer(self, audio):
        return np.random.randn(400, 87)

    # dp with optional silence
    def dp_optional_silence(self, post_probs, template):
        # i phone in template; j feats position
        # dp[i, j] = max(dp[i, j-1], dp[i-1, j-1], dp[i-2, j-1])
        feats_size = len(post_probs) ### frame size
        probs = np.array(post_probs)
        loss = np.zeros((len(template), feats_size))
        path = np.zeros((len(template), feats_size))

        for i in range(len(template)):   #### 看不懂？？？
            loss[0][i] = probs[0][template[i] - 1]

        # measure the silence phone encountered
        silences = 0
        for i in range(len(template)):
            for j in range(i, feats_size):
                if j == 0:
                    continue
                loss_skip = 0
                loss_go = 0
                loss_stay = loss[i][j - 1] + probs[j][template[i] - 1]
                if i > 0:
                    loss_go = loss[i - 1][j - 1] + probs[j][template[i] - 1]
                # if previous phone is silence
                # if the last phone of a word is the same as the first phone of its sub-sequent word, silence is fixed
                if i > 1 and template[i - 1] == 1 and template[i - 2] != template[i]:
                    loss_skip = loss[i - 2][j - 1] + probs[j][template[i] - 1]

                losses = [loss_stay, loss_go, loss_skip]
                loss[i][j] = max(losses)
                path[i][j] = losses.index(loss[i][j])
            if template[i] == 1:
                silences += 1

        template_position = len(template) - 1
        feat_position = feats_size - 1
        align_results = [template[template_position]]
        while True:
            if feat_position == 0:
                break
            if path[template_position][feat_position] == 0:
                align_results.append(template[template_position])
            elif path[template_position][feat_position] == 1:
                template_position -= 1
                align_results.append(template[template_position])
            else:
                template_position -= 2
                align_results.append(template[template_position])
            feat_position -= 1

        return align_results[::-1]

    def alignment(self, post_probs, t2p):
        try:
            aligned_result = self.dp_optional_silence(post_probs, t2p)
        except:
            print('error on this text!')

        return aligned_result

    def pscore(self, prob_segment, phone_id):
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

    def cal_fluency(self, result_dict, ar):
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

    def scores(self, audio, text):
        post_probs = self.kaldi_infer(audio)
        t2p = self.text2phone(text)
        aligned_result = self.alignment(post_probs, t2p)
        text = self.cleantext(text)
        probs = post_probs
        phones = self.phone_dict
        tphones = t2p
        reverse_phone = {}
        for key in phones.keys():
            reverse_phone[phones[key]] = key
        output = {}

        ar = aligned_result
        prob = probs
        template = tphones
        txt = text
        ps = tphones
        

        # construt result_dict
        words = txt.split(" ")
        result_dict = {}
        result_dict["words"] = []
        result_dict["text"] = txt
        template_position = 0
        align_position = 0

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
                phone_dict["score"] = self.pscore(prob[phone_dict["start"]:phone_dict["end"]], template[template_position])
                word_dict["phones"].append(phone_dict)
                template_position += 1
            vowels = []
            consonant = []
            for phone in word_dict["phones"]:
                if len(phone["phone"]) > 2:
                    vowels.append(phone["score"])
                else:
                    consonant.append(phone["score"])
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
        result_dict["pronunciation"] = round(phones_sum / phones_count, 4)
        result_dict["fluency"] = round(self.cal_fluency(result_dict, ar), 4)
        output = result_dict
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('lexiconaddr', type=str, help='the addr of lexicon')
    parser.add_argument('phonesaddr', type=str, help='the addr of phones')
    parser.add_argument('sr', type=int, help='sr')
    args = parser.parse_args()

    text = "how are you?"
    audio = 0

    Score_test = Score(args)
    score_output = Score_test.scores(audio, text)
