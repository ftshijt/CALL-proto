import json
import numpy as np
import math


phone_size = 86


# dp with optional silence
def dp_optional_silence(post_probs, template):
    # i phone in template; j feats position
    # dp[i, j] = max(dp[i, j-1], dp[i-1, j-1], dp[i-2, j-1])
    feats_size = len(post_probs)
    probs = np.array(post_probs)
    loss = np.zeros((len(template), feats_size))
    path = np.zeros((len(template), feats_size))
    # print(loss.shape)
    # print(probs.shape)
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

    print(align_results[::-1])
    print(template)
    return align_results[::-1]


# dp with fixed silence
def dp(post_probs, template):
    # i phone in template; j feats position
    # dp[i, j] = max(dp[i, j-1], dp[i -1, j-1])
    feats_size = len(post_probs)
    probs = np.array(post_probs)
    # probs = np.log(np.array(post_probs))
    loss = np.zeros((len(template), feats_size))
    path = np.zeros((len(template), feats_size))
    # print(loss.shape)
    # print(path.shape)
    # print(probs)
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


if __name__ ==  "__main__":
    # feature = json.load(open("feature.json", "r", encoding="utf-8"))
    t2p = json.load(open("call_2k/text/text2phone.json", "r", encoding="utf-8"))
    count = 0
    for i in range(31):
        print(i)
        aligned_result = {}
        feature = json.load(open("call_2k/feature/feature_%d.json"%i, "r", encoding="utf-8"))
        for key in feature.keys():
            try:
                # error processing for 33_03_00010
                if len(key) > 10:
                    aligned_result[key] = dp(feature[key], t2p[key[:7] + key[8:]])
                else:
                    aligned_result[key] = dp(feature[key], t2p[key])
            except:
                print(key)
            count += 1
            if count % 1000 == 0:
                print(count)
            # print(key)
        json.dump(aligned_result, open("call_2k/aligned/aligned_result_%d.json"%i, "w", encoding="utf-8"))


    aligned_result = {}
    # for key in feature:
    #     aligned_result[key] = dp(feature[key], t2p[key])
    #     print(key)
    # json.dump(aligned_result, open("aligned_result.json", "w", encoding="utf-8"))

    # for key in feature:
    #     aligned_result[key] = dp_optional_silence(feature[key], t2p[key])
    #     print(key)
    # json.dump(aligned_result, open("aligned_result.json", "w", encoding="utf-8"))
    # dp_optional_silence(feature["06_03_0021"], t2p["06_03_0021"])
