import re
import os
import csv
import math
import json
import string
import librosa
import argparse
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from kaldiio import load_ark

SILENCE = 1

def PackZero(integer, size):
    pack = size - len(str(integer))
    return "0" * pack + str(integer)


class Score:
    # Parameters introduction:
    # 'lexicon' is the address of lexicon file
    # 'phones' is the address of phones file
    # 'sr' is the sample rate of the audio
    # 'kaldi_workspace' is the address of kaldi's workspace
    # 'utt_id' is the user id of a specific example (i.e. one audio and the corresponding text)
    # 'non_stress_idx' is the idx of the phones with prominence which need to remove in non-prominence version
    # 'flag_version' is the flag of phone version: e.g. when flag_version=True, all the phones witn prominence will be removed.
    def __init__(
        self, lexicon, phones, sr, kaldi_workspace, utt_id, non_stress_idx, flag_version
    ):
        self.sr = sr
        self.reverse_phone_dict = {}
        self.vowels_id = non_stress_idx
        self.kaldi_workspace = kaldi_workspace
        self.utt_id = utt_id
        self.exp_path = "tdnn-XXX"
        self.flag_version = flag_version
        self.phone_dict, self.w2p = self.Word2Phone(lexicon, phones, flag_version)
        os.system("export PATH=%s:$PATH" % kaldi_workspace)

    def CleanText(self, text):
        punctuation_space = '.!,;:?"  '
        text = re.sub(r"[{}]+".format(punctuation_space), " ", text)
        return text.strip().upper()

    def Word2Phone(self, lexicon, phones, flag_version):
        w2p_file = open(lexicon, "r", encoding="latin-1")
        w2p_dict = {}
        phone_file = open(phones, "r", encoding="utf-8")
        phone_dict = {}
        reader = csv.reader(phone_file, delimiter=" ")
        id_num = 1
        if flag_version:
            counter = -1
        for line in reader:
            # if choose non-stress phone set, skip all stress phones
            if flag_version:
                counter += 1
                if counter in self.vowels_id:
                    continue
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
            if ";;;" in line[0]:
                continue
            else:
                word = line[0]
                phones = line[1].split(" ")
                # write non-stress phone into dict
                if flag_version:
                    translation = str.maketrans(
                        string.ascii_letters, string.ascii_letters, string.digits
                    )
                    phones_NoNum = []
                    for ph in phones:
                        if "0" in ph or "1" in ph or "2" in ph:
                            ph = ph.translate(translation)
                            phones_NoNum.append(ph)
                        else:
                            phones_NoNum.append(ph)

                    w2p_dict[word] = list(map(lambda x: phone_dict[x], phones_NoNum))
                else:
                    w2p_dict[word] = list(map(lambda x: phone_dict[x], phones))
        return phone_dict, w2p_dict

    def Text2Phone(self, text):
        text = self.CleanText(text)
        phone_list = [1]
        words = text.split(" ")

        self.reverse_phone_dict = {v: k for k, v in self.phone_dict.items()}

        for word in words:
            try:
                phone_list.extend(self.w2p[word])
                phone_list.append(1)
            except:
                print(word)
                print(text)
        t2p = phone_list
        print(t2p)
        return t2p

    def CreateTestEnv(self, audio, wav_id):
        # audio is a wavfile path
        data_path = os.path.join(
            self.kaldi_workspace, "data/audio_%s" % PackZero(self.utt_id, size=6)
        )
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        wav_file = os.path.join(data_path, "%s.wav" % wav_id)
        frame, sr = librosa.load(audio)
        frame, _ = librosa.effects.trim(frame)
        frame = librosa.resample(frame, sr, self.sr)
        sf.write(wav_file, frame, self.sr, subtype="PCM_16")

        wavscp = open(os.path.join(data_path, "wav.scp"), "w")
        utt2spk = open(os.path.join(data_path, "utt2spk"), "w")
        spk2utt = open(os.path.join(data_path, "spk2utt"), "w")
        text = open(os.path.join(data_path, "text"), "w")
        wavscp.write("%s_%s %s" % (wav_id, wav_id, wav_file))
        text.write("%s_%s %s" % (wav_id, wav_id, "placeholder"))
        utt2spk.write("%s_%s %s" % (wav_id, wav_id, wav_id))
        spk2utt.write("%s %s_%s" % (wav_id, wav_id, wav_id))
        wavscp.close()
        utt2spk.close()
        text.close()
        spk2utt.close()

        # create post template
        os.system(
            "cp -r %s %s"
            % (
                os.path.join(self.kaldi_workspace, "data/audio_template_post"),
                data_path + "_post",
            )
        )

    def KaldiInfer(self, audio):
        wav_id = PackZero(self.utt_id, size=6)
        self.CreateTestEnv(audio, wav_id)
        audio_path = "audio_%s" % PackZero(self.utt_id, size=6)
        # pass workspace, infer_set and num of jobs
        infer_log = os.popen(
            "%s %s %s 1"
            % (
                os.path.join(self.kaldi_workspace, "extract_post.sh"),
                self.kaldi_workspace,
                os.path.join(self.kaldi_workspace, "data", audio_path),
            )
        )
        infer_log = infer_log.readlines()
        if "infer success" not in " ".join(infer_log):
            print("Error\n%s" % infer_log)
        ark_post = os.path.join(
            self.kaldi_workspace, "data", audio_path + "_post", "phone_post.1.ark"
        )

        post_ark = load_ark(ark_post)
        for key, numpy_array in post_ark:
            if key == "%s_%s" % (wav_id, wav_id):
                post_numpy = numpy_array
                break
        self.utt_id += 1
        # add the column of stress into its non_stress phone
        if self.flag_version:
            del_list = [5, 9, 13, 17, 21, 25, 33, 37, 41, 48, 52, 62, 66, 76, 80]
            del_list_all = self.vowels_id
            for i in del_list:
                for j in [i, i - 1, i - 2]:
                    post_numpy[:, j - 1] += post_numpy[:, j]
            post_numpy = np.delete(post_numpy, del_list_all, axis=1)

        return post_numpy

    # dp with fixed silence which forces to assert silence between 2 phones
    def AlginFixedSilence(self, post_probs, template):
        # i phone in template; j feats position
        # dp[i, j] = max(dp[i, j-1], dp[i-1, j-1])
        feats_size = len(post_probs)
        probs = np.array(post_probs)
        loss = np.zeros((len(template), feats_size))
        path = np.zeros((len(template), feats_size))

        loss[0][0] = probs[0][0]

        for i in range(len(template)):
            for j in range(i, feats_size):
                if j == 0:
                    continue
                elif i == 0:
                    loss[i][j] = loss[i][j - 1] + probs[j][template[i] - 1]
                    # 0 for stay
                    path[i][j] = 0
                else:
                    loss_stay = loss[i][j - 1] + probs[j][template[i] - 1]
                    loss_go = loss[i - 1][j - 1] + probs[j][template[i] - 1]
                    if loss_go > loss_stay:
                        # 1 for change phone
                        path[i][j] = 1
                        loss[i][j] = loss_go
                    else:
                        # 0 for stay
                        path[i][j] = 0
                        loss[i][j] = loss_stay

        template_position = len(template) - 1
        feat_position = feats_size - 1
        align_results = [template[template_position]]
        while True:
            if template_position == 0 and feat_position == 0:
                break
            if path[template_position][feat_position] == 0:
                align_results.append(template[template_position])
            else:
                template_position -= 1
                align_results.append(template[template_position])
            feat_position -= 1
        return align_results[::-1]

    # dp with optional silence which has the option to skip the silence between 2 phones
    def AlginOptionalSilence(self, post_probs, template):
        # i phone in template; j feats position
        # dp[i, j] = max(dp[i, j-1], dp[i-1, j-1], dp[i-2, j-1])
        feats_size = len(post_probs)  # frame size
        probs = np.array(post_probs)
        loss = np.zeros((len(template), feats_size))
        path = np.zeros((len(template), feats_size))

        for i in range(len(template)):
            loss[0][i] = probs[0][template[i] - 1]


        # measure the silence phone encountered
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
                if i > 1 and template[i - 1] == SILENCE and template[i - 2] != template[i]:
                    loss_skip = loss[i - 2][j - 1] + probs[j][template[i] - 1]

                losses = [loss_stay, loss_go, loss_skip]
                loss[i][j] = max(losses)
                path[i][j] = losses.index(loss[i][j])

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

    def Align(self, post_probs, t2p):
        try:
            aligned_result_iner = self.AlginOptionalSilence(
                post_probs, t2p
            )  # Or use fixed silence: aligned_result_iner = self.AlginFixedSilence(post_probs, t2p)
        except:
            print("error on this text!")

        return aligned_result_iner

    def calcShannonEntropy(self, frame):  # in bit
        entropy = 0.0
        for x_value in frame:
            p = x_value
            log_p = np.log2(p)
            entropy -= p * log_p

        return entropy

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # inputs are the prob matrix and the id of the phone need to be read
    # outputs are the pronoun score and the id of the phone actually read
    def PScore_GOP(self, prob_segment, phone_id):
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1

        if len(prob_segment) == 0:
            return 0
        score_prob = prob_segment
        target_result = []
        for prob in score_prob:
            target_result.append(prob[phone_id - 1] * 100)
        target_result = round(sum(target_result) / len(target_result), 4)
        return target_result, phone_read

    # inputs are the prob matrix and the id of the phone need to be read
    # outputs are the pronoun score and the id of the phone actually read
    def PScore_sGOP(self, prob_segment, phone_id):
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1

        if len(prob_segment) == 0:
            return 0
        length = len(prob_segment)
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
                target_result.append(
                    math.log(max(prob)) / math.log(prob[phone_id - 1]) * 100
                )
        target_result = round(sum(target_result) / len(target_result), 4)
        return target_result, phone_read

    # inputs are the prob matrix and the id of the phone need to be read
    # outputs are the pronoun score and the id of the phone actually read
    def PScore_TAGOP(self, prob_segment, phone_id):
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1

        if len(prob_segment) == 0:
            return 0
        length = len(prob_segment)

        # Entropy part
        entropy_list = []
        for i in range(length):
            ent = self.calcShannonEntropy(prob_segment[i])
            entropy_list.append(ent)

        # reciprocal weighted
        score_prob = prob_segment
        entropy_list_reciprocal = list(
            1 / entropy_list[i] for i in range(len(entropy_list))
        )

        target_result = []
        for idx, prob in enumerate(score_prob):
            if prob[phone_id - 1] > 0.5:
                target_result.append(
                    100 * (entropy_list_reciprocal[idx]) / sum(entropy_list_reciprocal)
                )
            else:
                if prob[phone_id - 1] < 1e-8:
                    target_result.append(0)
                    continue
                target_result.append(
                    math.log(max(prob))
                    / math.log(prob[phone_id - 1])
                    * 100
                    * (entropy_list_reciprocal[idx])
                    / sum(entropy_list_reciprocal)
                )
        target_result = round(sum(target_result), 4)
        return target_result, phone_read

    # outputs are the pronoun score and the id of the phone actually read
    def PScore_pGOP(
        self, prob_segment, phone_id, idx_1, idx_2, beta
    ):  # with only Duration Factor w/o Transition Factor
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1

        if len(prob_segment) == 0:
            return 0
        score_prob = prob_segment
        target_result = []
        for prob in score_prob:
            target_result.append(prob[phone_id - 1] * 100)
        target_result = round(sum(target_result) / len(target_result), 4)

        # Duration Factor:
        real = real_matrix[idx_1][idx_2]
        standard = standard_matrix[idx_1][idx_2]
        mean_phone = float(phone_mean_std_dict[str(phone_id)]["mean"])
        std_phone = float(phone_mean_std_dict[str(phone_id)]["std"])
        coeff_x = 1.5
        thres = mean_phone + coeff_x * std_phone
        delta = abs(real - standard) - thres
        p_score = target_result - beta * delta * target_result
        if p_score > 100:
            p_score = 100
        if p_score < 0:
            p_score = 0
        return p_score, phone_read

    # outputs are the pronoun score and the id of the phone actually read
    def PScore_CaGOP(self, prob_segment, phone_id, idx_1, idx_2, beta):
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1

        if len(prob_segment) == 0:
            return 0
        length = len(prob_segment)

        # Entropy part
        entropy_list = []
        for i in range(length):
            ent = self.calcShannonEntropy(prob_segment[i])
            entropy_list.append(ent)

        # reciprocal weighted
        score_prob = prob_segment
        entropy_list_reciprocal = list(
            1 / entropy_list[i] for i in range(len(entropy_list))
        )

        target_result = []
        for idx, prob in enumerate(score_prob):
            if prob[phone_id - 1] > 0.5:
                target_result.append(
                    100 * (entropy_list_reciprocal[idx]) / sum(entropy_list_reciprocal)
                )
            else:
                if prob[phone_id - 1] < 1e-8:
                    target_result.append(0)
                    continue
                target_result.append(
                    math.log(max(prob))
                    / math.log(prob[phone_id - 1])
                    * 100
                    * (entropy_list_reciprocal[idx])
                    / sum(entropy_list_reciprocal)
                )
        target_result = round(sum(target_result), 4)

        # Duration Factor:
        real = real_matrix[idx_1][idx_2]
        standard = standard_matrix[idx_1][idx_2]
        mean_phone = float(phone_mean_std_dict[str(phone_id)]["mean"])
        std_phone = float(phone_mean_std_dict[str(phone_id)]["std"])
        coeff_x = 1.5
        thres = mean_phone + coeff_x * std_phone
        delta = abs(real - standard) - thres
        p_score = target_result - beta * delta * target_result
        if p_score > 100:
            p_score = 100
        if p_score < 0:
            p_score = 0

        return p_score, phone_read

    def CalcFluency(self, result_dict, ar):
        if len(result_dict["words"]) == 1:
            return 100
        content = result_dict["end"] - result_dict["start"]
        silence_sum = 0
        for p in ar[result_dict["start"] : result_dict["end"]]:
            if p == 1:
                silence_sum += 1
        standard_size = 3 * len(result_dict["words"]) - 3
        silence_sum = max(silence_sum - standard_size, 0)
        fluency = max(1 - silence_sum / (content - silence_sum), 0) * 100

        return fluency

    # input parameters introduction:
    # 'audio' is the address of the audio
    # 'text' is the corresponding text (not address!)
    # 'method_name' is the method to use in pronoun score (i.e. GOP, sGOP, TAGOP, pGOP, CaGOP)
    # 'beta' is one paremeter needed in duration factor calculation
    def CalcScores(self, audio, text, method_name, read_idx, beta):
        post_probs = self.KaldiInfer(audio)
        t2p = self.Text2Phone(text)
        aligned_result = self.Align(post_probs, t2p)
        text = self.CleanText(text)
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
            idx_2 = 0
            while template_position < len(template):

                # continued phones - same phone output
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
                while (
                    align_position < len(ar)
                    and ar[align_position] == template[template_position]
                ):
                    align_position += 1
                phone_dict["end"] = align_position

                try:
                    if method_name == "GOP":
                        phone_dict["score"], phone_read = self.PScore_GOP(
                            prob[phone_dict["start"] : phone_dict["end"]],
                            template[template_position],
                        )
                    if method_name == "sGOP":
                        phone_dict["score"], phone_read = self.PScore_sGOP(
                            prob[phone_dict["start"] : phone_dict["end"]],
                            template[template_position],
                        )
                    if method_name == "TAGOP":
                        phone_dict["score"], phone_read = self.PScore_TAGOP(
                            prob[phone_dict["start"] : phone_dict["end"]],
                            template[template_position],
                        )
                    if method_name == "pGOP":
                        phone_dict["score"], phone_read = self.PScore_pGOP(
                            prob[phone_dict["start"] : phone_dict["end"]],
                            template[template_position],
                            read_idx,
                            idx_2,
                            beta,
                        )
                    if method_name == "CaGOP":
                        phone_dict["score"], phone_read = self.PScore_CaGOP(
                            prob[phone_dict["start"] : phone_dict["end"]],
                            template[template_position],
                            read_idx,
                            idx_2,
                            beta,
                        )

                except AttributeError:
                    print("The method name is invalid!")
                    break

                # show each phones to be read
                if len(phone_dict["phone"]) > 2:
                    if (
                        len(str(reverse_phone[phone_read])) > 2
                        and phone_dict["phone"][:-1]
                        == str(reverse_phone[phone_read])[:-1]
                    ):
                        proun_eval = (
                            phone_dict["phone"]
                            + " was read as "
                            + str(reverse_phone[phone_read])
                        )
                    if (
                        len(str(reverse_phone[phone_read])) > 2
                        and phone_dict["phone"][:-1]
                        != str(reverse_phone[phone_read])[:-1]
                    ):
                        proun_eval = (
                            phone_dict["phone"][:-1]
                            + " was read as "
                            + str(reverse_phone[phone_read])[:-1]
                        )
                    if len(str(reverse_phone[phone_read])) <= 2:
                        proun_eval = (
                            phone_dict["phone"][:-1]
                            + " was read as "
                            + str(reverse_phone[phone_read])
                        )
                if len(phone_dict["phone"]) <= 2:
                    if len(str(reverse_phone[phone_read])) > 2:
                        proun_eval = (
                            phone_dict["phone"]
                            + " was read as "
                            + str(reverse_phone[phone_read])[:-1]
                        )
                    if len(str(reverse_phone[phone_read])) <= 2:
                        proun_eval = (
                            phone_dict["phone"]
                            + " was read as "
                            + str(reverse_phone[phone_read])
                        )

                phone_dict["Evaluation"] = proun_eval
                word_dict["phones"].append(phone_dict)
                template_position += 1
                idx_2 += 1
            vowels = []
            consonant = []
            for phone in word_dict["phones"]:
                if len(phone["phone"]) > 2:
                    vowels.append(phone["score"])
                else:
                    consonant.append(phone["score"])
            word_dict["score"] = round(
                (sum(vowels) + sum(consonant)) / (len(vowels) + len(consonant)), 4
            )
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
        result_dict["integrity"] = round(
            inter_word / len(result_dict["words"]) * 100, 4
        )
        result_dict["start"] = (result_dict["words"])[0]["start"]
        result_dict["end"] = (result_dict["words"])[-1]["end"]
        result_dict["pronunciation"] = round(phones_sum / phones_count, 4)
        result_dict["fluency"] = round(self.CalcFluency(result_dict, ar), 4)
        output = result_dict
        print(output)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lexiconaddr", type=str, help="the addr of lexicon")
    parser.add_argument("phonesaddr", type=str, help="the addr of phones")
    parser.add_argument("sr", type=int, help="sample rate")
    parser.add_argument("kaldi_workspace", type=str, help="kaldi_workspace")
    parser.add_argument("utt_id", type=int, help="user_id")
    parser.add_argument(
        "method_name",
        type=str,
        choices=["CaGOP", "TAGOP", "pGOP", "sGOP", "GOP"],
        help="GOP method to use, e.g. CaGOP",
    )
    parser.add_argument(
        "--version",
        type=bool,
        default=False,
        help="Use stress or not, typing True or False",
    )
    args = parser.parse_args()

    with open("non_stress.json", "r") as f:
        non_stress_dict = json.load(f)

    # parameters
    beta = 1
    read_idx = 1
    non_stress_idx = non_stress_dict["idx"]

# Below is an example
"""
   # text = 'Did you hear that Daniel made the cut to Varsity Track and Field?'
   # text = "I have to study for my AP Calculus test."
   # text = "He scored three amazing goals in yesterday's game."
    text =  "Do you know where the Emergency Room is located?"
   # text = "hello, how are you doing?"

    audio = '/home/nan/CALL-proto/Fun-emes/django_project/microphone-results_4.wav'
    Score_test = Score(args.lexiconaddr, args.phonesaddr, args.sr, args.kaldi_workspace, args.utt_id, non_stress_idx, args.version)
    score_output = Score_test.CalcScores(audio, text, args.method_name, read_idx, beta)
"""
