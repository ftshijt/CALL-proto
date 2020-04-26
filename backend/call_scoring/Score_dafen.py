import json
import re
import os
import csv
import numpy as np
import math
import librosa
import soundfile as sf
import argparse
from scipy.io import wavfile
from kaldiio import load_ark
import string 

def PackZero(integer, size):
        pack = size - len(str(integer))
        return "0" * pack + str(integer)


class Score():
    def __init__(self, lexicon, phones, sr, kaldi_workspace, utt_id, flag_version=False):
        self.phone_dict, self.w2p = self.Word2Phone(lexicon, phones, flag_version)
        self.sr = sr
        self.kaldi_workspace = kaldi_workspace
        self.utt_id = utt_id
        self.utt_id_fix = str(utt_id)
        self.exp_path = "tdnn-XXX"
        self.flag_version = flag_version
        os.system("export PATH=%s:$PATH" % kaldi_workspace)
        # ...

    def CleanText(self, text):
        punctuation = '.!,;:?"'
        Space = '  '
        text = re.sub(r'[{}]+'.format(punctuation),' ',text)
        text = re.sub(r'[{}]+'.format(Space),' ',text)
        return text.strip().upper()

    def Word2Phone(self, lexicon, phones, flag_version):
        w2p_file = open(lexicon, "r", encoding="latin-1")
        w2p_dict = {}
        phone_file = open(phones, "r", encoding="utf-8")
        phone_dict = {}
        reader = csv.reader(phone_file, delimiter=" ")
        id_num = 1
        if flag_version:
            counter = -1  ### Nan
        for line in reader:
            ### Nan
            if flag_version:
               counter += 1
               if counter == 3 or counter == 4 or counter == 5:
                   continue
               if counter == 7 or counter == 8 or counter == 9:
                   continue
               if counter == 11 or counter == 12 or counter == 13:
                   continue
               if counter == 15 or counter == 16 or counter == 17:
                   continue
               if counter == 19 or counter == 20 or counter == 21:
                   continue
               if counter == 23 or counter == 24 or counter == 25:
                   continue
               if counter == 31 or counter == 32 or counter == 33:
                   continue
               if counter == 35 or counter == 36 or counter == 37:
                   continue
               if counter == 39 or counter == 40 or counter == 41:
                   continue
               if counter == 46 or counter == 47 or counter == 48:
                   continue
               if counter == 50 or counter == 51 or counter == 52:
                   continue
               if counter == 60 or counter == 61 or counter == 62:
                   continue
               if counter == 64 or counter == 65 or counter == 66:
                   continue
               if counter == 74 or counter == 75 or counter == 76:
                   continue
               if counter == 78 or counter == 79 or counter == 80:
                   continue
               ###
            phone_dict[line[0]] = int(id_num)
            id_num += 1
        print (phone_dict)
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

                #### Nan
                if flag_version:
                   translation = str.maketrans(string.ascii_letters, string.ascii_letters, string.digits)
                   phones_NoNum = []
                   for ph in phones:
                      if '0' in ph or '1' in ph or '2' in ph:
                         ph = ph.translate(translation)
                         phones_NoNum.append(ph)
                      else:
                         phones_NoNum.append(ph)
                ####
                   w2p_dict[word] = list(map(lambda x: phone_dict[x], phones_NoNum))
                else: 
                   w2p_dict[word] = list(map(lambda x: phone_dict[x], phones))
        return phone_dict, w2p_dict


    def Text2Phone(self, text):
        text = self.CleanText(text)
        phone_list = [1]
        words = text.split(" ")

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
        data_path = os.path.join(self.kaldi_workspace, "data/audio_%s"%PackZero(self.utt_id, size=8))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        wav_file = os.path.join(data_path, "%s.wav"%wav_id)
        frame, sr = librosa.load(audio)
        frame, _ = librosa.effects.trim(frame)
        frame = librosa.resample(frame, sr, self.sr)
        sf.write(wav_file, frame, self.sr, subtype='PCM_16')

        wavscp = open(os.path.join(data_path, "wav.scp"), "w")
        utt2spk = open(os.path.join(data_path, "utt2spk"), "w")
        spk2utt = open(os.path.join(data_path, "spk2utt"), "w")
        text = open(os.path.join(data_path, "text"), "w")
        wavscp.write("%s_%s %s" %(wav_id, wav_id, wav_file))
        text.write("%s_%s %s" %(wav_id, wav_id, "placeholder"))
        utt2spk.write("%s_%s %s" %(wav_id, wav_id, wav_id))
        spk2utt.write("%s %s_%s" %(wav_id, wav_id, wav_id))
        wavscp.close()
        utt2spk.close()
        text.close()
        spk2utt.close()

        # create post template
        os.system("cp -r %s %s" %(os.path.join(self.kaldi_workspace, "data/audio_template_post"), 
            data_path + "_post"))

    def KaldiInfer(self, audio):
        wav_id = PackZero(self.utt_id, size=8)
        self.CreateTestEnv(audio, wav_id)
        audio_path = "audio_%s"%PackZero(self.utt_id, size=8)
        # pass workspace, infer_set and num of jobs
        infer_log = os.popen("%s %s %s 1" 
            % (os.path.join(self.kaldi_workspace, "extract_post.sh"),
                self.kaldi_workspace, os.path.join(self.kaldi_workspace, "data", audio_path)))
        infer_log = infer_log.readlines()
        if "infer success" not in " ".join(infer_log):
            print("Error\n%s" %infer_log)
        ark_post = os.path.join(self.kaldi_workspace,
            "data", audio_path + "_post", "phone_post.1.ark")
       ###############
      #  ark_post = os.path.join('/home/nan/CALL-proto/backend/audio_000151_post_test/phone_post.1.ark')

        post_ark = load_ark(ark_post)
        for key, numpy_array in post_ark:
            if key == "%s_%s" %(wav_id, wav_id):
                post_numpy = numpy_array
                break
       # print(post_numpy.shape)
        self.utt_id += 1
        ### Nan
        if self.flag_version:
           del_list = [5,9,13,17,21,25,33,37,41,48,52,62,66,76,80]
           del_list_all = [3,4,5,7,8,9,11,12,13,15,16,17,19,20,21,23,24,25,31,32,33,35,36,37,39,40,41,46,47,48,50,51,52,60,61,62,64,65,66,74,75,76,78,79,80]
           for i in del_list:
                for j in [i,i-1,i-2]:
                    post_numpy[:,j-1] += post_numpy[:,j]
           post_numpy = np.delete(post_numpy, del_list_all, axis=1)
           np.savetxt('post_new.csv', post_numpy, delimiter = ',')     ### Delete later
        ###
        return post_numpy


    # dp with fixed silence
    def AlginFixedSilence(self, post_probs, template):
        # i phone in template; j feats position
        # dp[i, j] = max(dp[i, j-1], dp[i -1, j-1])
        feats_size = len(post_probs)
        probs = np.array(post_probs)
        # probs = np.log(np.array(post_probs))
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


    # dp with optional silence
    def AlginOptionalSilence(self, post_probs, template):
        # i phone in template; j feats position
        # dp[i, j] = max(dp[i, j-1], dp[i-1, j-1], dp[i-2, j-1])
        feats_size = len(post_probs) # frame size
        probs = np.array(post_probs)
        loss = np.zeros((len(template), feats_size))
        path = np.zeros((len(template), feats_size))

        for i in range(len(template)):
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

    def Align(self, post_probs, t2p, align_file):
        try:
            aligned_result_iner = self.AlginOptionalSilence(post_probs, t2p)
          #  aligned_result_iner = self.AlginFixedSilence(post_probs, t2p)
        except:
            print('error on this text!')
        aligned_result_iner_str = list(map(str, aligned_result_iner))
        aligned_result_str = " ".join(aligned_result_iner_str)+"\n"
        align_file.write(aligned_result_str)
      #  aligned_result_iner = self.AlginOptionalSilence_v2(post_probs, t2p)
      #  aligned_result_iner = self.AlginFixedSilence(post_probs, t2p)
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

    def PScore_GOP(self, prob_segment, phone_id):
        ### Nan
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1
        ###
        if len(prob_segment) == 0:
            return 0
        entropy_list = None
        score_prob = prob_segment
        target_result = []
        for prob in score_prob:
            if prob[phone_id - 1] > 0.5:
                target_result.append(100)
            else:
                if prob[phone_id - 1] < 1e-8:
                    target_result.append(0)
                    continue
                target_result.append(math.log(max(prob)) / math.log(prob[phone_id-1]) * 100)
        target_result = sorted(target_result)
        return round(sum(target_result[:min(1, len(target_result))]) / min(len(target_result), 1), 4), phone_read, entropy_list


    def PScore_sGOP(self, prob_segment, phone_id):
        ### Nan
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1
        ###
        if len(prob_segment) == 0:
            return 0
        entropy_list = None
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
                target_result.append(math.log(max(prob)) / math.log(prob[phone_id-1]) * 100)
        target_result = sorted(target_result)
        return round(sum(target_result[:min(1, len(target_result))]) / min(len(target_result), 1), 4), phone_read, entropy_list


    def PScore_EntGOP(self, prob_segment, phone_id):
        ### Nan
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1
        ###
        if len(prob_segment) == 0:
            return 0
        length =len(prob_segment)
        ### Nan: Entropy part
        entropy_list = []
        for i in range(length):
            ent = self.calcShannonEntropy(prob_segment[i])
            entropy_list.append(ent)
        MaxEnt_ind = entropy_list.index(min(entropy_list))
        if (MaxEnt_ind - math.ceil(length / 4)) >= 0:
            if (MaxEnt_ind + math.ceil(length / 4)) <= length:
                score_start = MaxEnt_ind - math.ceil(length / 4)
                score_end = MaxEnt_ind + math.ceil(length / 4)
            else:
                score_start = math.ceil(length / 2)
                score_end = length
        else:
            score_start = 0
            if (math.ceil(length / 2)) <= length:
                score_end = math.ceil(length / 2)
            else:
                score_end = length

        ####
        score_prob = prob_segment[score_start:score_end]
        target_result = []
        for prob in score_prob:
            if prob[phone_id - 1] > 0.5:
                target_result.append(100)
            else:
                if prob[phone_id - 1] < 1e-8:
                    target_result.append(0)
                    continue
                target_result.append(math.log(max(prob)) / math.log(prob[phone_id-1]) * 100)
        target_result = sorted(target_result)
        return round(sum(target_result[:min(1, len(target_result))]) / min(len(target_result), 1), 4), phone_read, entropy_list


    def PScore_wGOP_linear(self, prob_segment, phone_id):
        ### Nan
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1
        ###
        if len(prob_segment) == 0:
            return 0
        length =len(prob_segment)
        ### linear
        score_prob = prob_segment
        phones_prob = np.sum(prob_segment, axis=0) / length
        for phone in range(len(phones_prob)):
            weight_phone = 0
            for i in (list(range(phone)) + list(range(phone+1, len(phones_prob)))):
                weight_phone += phones_prob[phone] / (phones_prob[phone] + phones_prob[i])
            weight_phone = weight_phone / (len(phones_prob)-1)
            score_prob[:, phone] = weight_phone * score_prob[:, phone]
        ###
        target_result = []
        for prob in score_prob:
            if prob[phone_id - 1] > 0.5:
                target_result.append(100)
            else:
                if prob[phone_id - 1] < 1e-8:
                    target_result.append(0)
                    continue
                target_result.append(math.log(max(prob)) / math.log(prob[phone_id-1]) * 100)
        target_result = sorted(target_result)
        entropy_list = None
        return round(sum(target_result[:min(1, len(target_result))]) / min(len(target_result), 1), 4), phone_read, entropy_list


    def PScore_wGOP_softmax(self, prob_segment, phone_id):
        ### Nan
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1
        ###
        if len(prob_segment) == 0:
            return 0
        length =len(prob_segment)
        ### Nan: Entropy part
        entropy_list = []
        for i in range(length):
            ent = self.calcShannonEntropy(prob_segment[i])
            entropy_list.append(ent)
        ### SoftMax
        SoftMax_list = self.softmax(entropy_list)
        score_prob = prob_segment
        for i in range(length):
            score_prob[i] = score_prob[i] / SoftMax_list[i]
        ###
        target_result = []
        for prob in score_prob:
            if prob[phone_id - 1] > 0.5:
                target_result.append(100)
            else:
                if prob[phone_id - 1] < 1e-8:
                    target_result.append(0)
                    continue
                target_result.append(math.log(max(prob)) / math.log(prob[phone_id-1]) * 100)
        target_result = sorted(target_result)
        return round(sum(target_result[:min(1, len(target_result))]) / min(len(target_result), 1), 4), phone_read, entropy_list


    def PScore_wGOP_reciprocal(self, prob_segment, phone_id):
        ### Nan
        prob_segment_log = np.log(np.array(prob_segment))
        prob_segment_logSum = np.sum(prob_segment_log, axis=0)
        phone_read = np.argmax(prob_segment_logSum) + 1
        ###
        if len(prob_segment) == 0:
            return 0
        length =len(prob_segment)
        ### Nan: Entropy part
        entropy_list = []
        for i in range(length):
            ent = self.calcShannonEntropy(prob_segment[i])
            entropy_list.append(ent)
        ### reciprocal
        score_prob = prob_segment
        for i in range(length):
            score_prob[i] = score_prob[i] / entropy_list[i]
        ###
        target_result = []
        for prob in score_prob:
            if prob[phone_id - 1] > 0.5:
                target_result.append(100)
            else:
                if prob[phone_id - 1] < 1e-8:
                    target_result.append(0)
                    continue
                target_result.append(math.log(max(prob)) / math.log(prob[phone_id-1]) * 100)
        target_result = sorted(target_result)
        return round(sum(target_result[:min(1, len(target_result))]) / min(len(target_result), 1), 4), phone_read, entropy_list



    def CalcFluency(self, result_dict, ar):
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

    def CalcScores(self, audio, text, align_file):
        post_probs = self.KaldiInfer(audio)
        t2p = self.Text2Phone(text)
        aligned_result = self.Align(post_probs, t2p, align_file)
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
        ### Nan: save Entropy_list
        Ent_file = open('/home/nan/CALL-proto/backend/call_scoring/Ent_Dafen/Entropy_%s.txt'%self.utt_id_fix, 'w')
        Ent_list_all = []
        ###
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
                phone_dict["score"], phone_read, entropy_list = self.PScore_wGOP_linear(prob[phone_dict["start"]:phone_dict["end"]], template[template_position])

                if entropy_list == None:
                    Ent_list_all = None
                else:
                    entropy_list.insert(0, phone_dict["phone"]) 
                    entropy_list.insert(0, word_dict["word"]) 
               # if '0' in phone_dict["phone"] or '1' in phone_dict["phone"] or '2' in phone_dict["phone"]:
               ### Nan
                if entropy_list != None:
                    Ent_list_all.append(entropy_list)
                if len(phone_dict["phone"]) > 2:
                    if len(str(reverse_phone[phone_read])) > 2 and phone_dict["phone"][:-1] == str(reverse_phone[phone_read])[:-1]:
                        proun_eval = phone_dict["phone"] + ' was read as ' + str(reverse_phone[phone_read])
                    if len(str(reverse_phone[phone_read])) > 2 and phone_dict["phone"][:-1] != str(reverse_phone[phone_read])[:-1]:
                        proun_eval = phone_dict["phone"][:-1] + ' was read as ' + str(reverse_phone[phone_read])[:-1]
                    if len(str(reverse_phone[phone_read])) <= 2:
                        proun_eval = phone_dict["phone"][:-1] + ' was read as ' + str(reverse_phone[phone_read])
                if len(phone_dict["phone"]) <= 2:
                    if len(str(reverse_phone[phone_read])) > 2:
                        proun_eval = phone_dict["phone"] + ' was read as ' + str(reverse_phone[phone_read])[:-1]
                    if len(str(reverse_phone[phone_read])) <= 2:
                        proun_eval = phone_dict["phone"] + ' was read as ' + str(reverse_phone[phone_read])
               ###
                phone_dict["Evaluation"] = proun_eval
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
        ### Nan: save Entropy_list
        if Ent_list_all != None:
            for ent_list in Ent_list_all:
                Ent_str = list(map(str, ent_list))
                Ent_str_t = "\t".join(Ent_str)
                Ent_file.write(Ent_str_t)
                Ent_file.write('\n')
        Ent_file.close() 
        ###
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
        result_dict["fluency"] = round(self.CalcFluency(result_dict, ar), 4)
        output = result_dict
        print(output)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('lexiconaddr', type=str, help='the addr of lexicon')
    parser.add_argument('phonesaddr', type=str, help='the addr of phones')
    parser.add_argument('sr', type=int, help='sr')
    parser.add_argument('kaldi_workspace', type=str, help='kaldi_workspace')
   # parser.add_argument('utt_id', type=int, help='utt_id')
    parser.add_argument('--version', type=bool, default = False, help='Version 1 or 2')
    args = parser.parse_args()

   # text = 'Did you hear that Daniel made the cut to Varsity Track and Field?'
   # text = "I have to study for my AP Calculus test."
   # text = "He scored three amazing goals in yesterday's game."
   # text =  "Do you know where the Emergency Room is located?"
   # text = "hello, how are you doing?"

   # audio = '/home/nan/CALL-proto/Fun-emes/django_project/microphone-results_5.wav'
    score_path = "scor_wGOP_linear.csv"
    align_file = "Call_DaFen.ali"
    align_file = open(align_file, "w")

    with open(score_path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['ID', 'accuracy', 'fluency', 'integrity']
        csv_write.writerow(csv_head)

    for num in (list(range(30,35)) + list(range(40,48))):
   # for num in range(44,48):
        num_str = str(num)
        text_all = []
        texts = open("/home/nan/CALL-proto/Fun-emes/django_project/CALL_data_df/" + num_str + '/text.txt',"r")
        lines = texts.readlines()
        num_text = len(lines)
        for line in lines:
            text_all.append(line)
        person = 1
        counter_id = 0 ### in case of lacking wav in certain people
        while True:
            person_str = str(person)
            try:
                
                for wav in range(1,num_text+1):
                    wav_str = str(wav)
                    for i in range(4-len(wav_str)):
                        wav_str = '0' + wav_str
                    wav_str = wav_str + '.wav'
                    audio = "/home/nan/CALL-proto/Fun-emes/django_project/CALL_data_df/" + num_str + '/' + person_str + '/' + wav_str
                    text = text_all[wav-1]
                    utt_id = num_str + '0' + person_str + '0' + str(wav)
                    if len(utt_id) < 6:
                        for j in range(6-len(utt_id)):
                            utt_id = '0' + utt_id
                    utt_id = int(utt_id)
                    num_str_disp = num_str
                    person_str_disp = person_str
                    if len(num_str) < 2:
                        num_str_disp = '0' + num_str
                    if len(person_str) < 2:
                         person_str_disp = '0' + person_str
                    score_ID = num_str_disp + '_' + person_str_disp + '_' + wav_str[0:4]
                    align_file.write(score_ID + "\t")
                   # print (counter_id)
                    Score_test = Score(args.lexiconaddr, args.phonesaddr, args.sr, args.kaldi_workspace, utt_id, args.version)
                    if counter_id < 5:
                        try:
                            score_output = Score_test.CalcScores(audio, text, align_file)
                        except FileNotFoundError:
                            counter_id += 1
                            continue
                    else:
                        score_output = Score_test.CalcScores(audio, text, align_file)
                    wav_id = utt_id
                    json.dump(score_output, open("/home/nan/CALL-proto/Fun-emes/django_project/DaFen/score_wav_%s.json"%wav_id, "w", encoding="utf-8"))
                    score_integrity = score_output['integrity']
                    score_pronunciation = score_output['pronunciation']
                    score_fluency = score_output['fluency']
                   # path = "score.csv"
                    with open(score_path,'a+') as f:
                        csv_write = csv.writer(f)
                        data_row = [score_ID, str(score_pronunciation), str(score_fluency), str(score_integrity)]
                        csv_write.writerow(data_row)
            except FileNotFoundError:
                break

            person += 1

    align_file.close() 

