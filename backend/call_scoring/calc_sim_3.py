import scipy
import scipy.stats
import numpy as np
from minepy import MINE
import minepy
import csv


with open('rater1.tsv', encoding = 'UTF-8') as rater1:
    rater1_pron = []
    rater1_fluency = []
    rater1_integrity = []
    IDs_1 = []
    rater1_reader = csv.reader(rater1)
    for row in rater1_reader:
        rater1_pron.append(row[0].split('\t')[3])
        rater1_fluency.append(row[0].split('\t')[2])
        rater1_integrity.append(row[0].split('\t')[1])
        IDs_1.append(row[0])
    
    rater1_pron = list(map(float,rater1_pron[1:]))
    rater1_fluency = list(map(float,rater1_fluency[1:]))
    rater1_integrity = list(map(float,rater1_integrity[1:]))
    IDs_1 = IDs_1[1:]


with open('rater2.csv') as rater2:
    rater2_pron = []
    rater2_fluency = []
    rater2_integrity = []
    IDs_2 = []
    rater2_reader = csv.reader(rater2)
    for row in rater2_reader:
        rater2_pron.append(row[1])
        rater2_fluency.append(row[2])
        rater2_integrity.append(row[3])
        IDs_2.append(row[0])

    rater2_pron = list(map(float,rater2_pron[1:]))
    rater2_fluency = list(map(float,rater2_fluency[1:]))
    rater2_integrity = list(map(float,rater2_integrity[1:]))
    IDs_2 = IDs_2[1:]


with open('rater3.tsv') as rater3:
    rater3_pron = []
    rater3_fluency = []
    rater3_integrity = []
    IDs_3 = []
    rater3_reader = csv.reader(rater3)
    for row in rater3_reader:
        rater3_pron.append(row[0].split('\t')[1])
        rater3_fluency.append(row[0].split('\t')[2])
        rater3_integrity.append(row[0].split('\t')[3])
        IDs_3.append(row[0].split('\t')[0])

    rater3_pron = list(map(float,rater3_pron[1:]))
    rater3_fluency = list(map(float,rater3_fluency[1:]))
    rater3_integrity = list(map(float,rater3_integrity[1:]))
    IDs_3 = IDs_3[1:]


with open('score_fixTAGOP.csv') as fixTAGOP:
    fixTAGOP_pron = []
    fixTAGOP_fluency = []
    fixTAGOP_integrity = []
    IDs = []
    fixTAGOP_reader = csv.reader(fixTAGOP)
    for row in fixTAGOP_reader:
        fixTAGOP_pron.append(row[1])
        fixTAGOP_fluency.append(row[2])
        fixTAGOP_integrity.append(row[3])
        IDs.append(row[0])

    fixTAGOP_pron = list(map(float,fixTAGOP_pron[1:]))
    fixTAGOP_fluency = list(map(float,fixTAGOP_fluency[1:]))
    fixTAGOP_integrity = list(map(float,fixTAGOP_integrity[1:]))
    IDs = IDs[1:]

    fixTAGOP_pron = list(fixTAGOP_pron[i]/10 for i in range(len(fixTAGOP_pron)))
    fixTAGOP_fluency = list(fixTAGOP_fluency[i]/10 for i in range(len(fixTAGOP_fluency)))
    fixTAGOP_integrity = list(fixTAGOP_integrity[i]/10 for i in range(len(fixTAGOP_integrity)))

with open('score_fixTAGOP_weighted.csv') as fixTAGOP_weighted:
    fixTAGOP_weighted_pron = []
    fixTAGOP_weighted_fluency = []
    fixTAGOP_weighted_integrity = []
    IDs = []
    fixTAGOP_weighted_reader = csv.reader(fixTAGOP_weighted)
    for row in fixTAGOP_weighted_reader:
        fixTAGOP_weighted_pron.append(row[1])
        fixTAGOP_weighted_fluency.append(row[2])
        fixTAGOP_weighted_integrity.append(row[3])
        IDs.append(row[0])

    fixTAGOP_weighted_pron = list(map(float,fixTAGOP_weighted_pron[1:]))
    fixTAGOP_weighted_fluency = list(map(float,fixTAGOP_weighted_fluency[1:]))
    fixTAGOP_weighted_integrity = list(map(float,fixTAGOP_weighted_integrity[1:]))
    IDs = IDs[1:]

    fixTAGOP_weighted_pron = list(fixTAGOP_weighted_pron[i]/10 for i in range(len(fixTAGOP_weighted_pron)))
    fixTAGOP_weighted_fluency = list(fixTAGOP_weighted_fluency[i]/10 for i in range(len(fixTAGOP_weighted_fluency)))
    fixTAGOP_weighted_integrity = list(fixTAGOP_weighted_integrity[i]/10 for i in range(len(fixTAGOP_weighted_integrity)))


with open('score_GOP.csv') as GOP:
    GOP_pron = []
    GOP_fluency = []
    GOP_integrity = []
    IDs = []
    GOP_reader = csv.reader(GOP)
    for row in GOP_reader:
        GOP_pron.append(row[1])
        GOP_fluency.append(row[2])
        GOP_integrity.append(row[3])
        IDs.append(row[0])

    GOP_pron = list(map(float,GOP_pron[1:]))
    GOP_fluency = list(map(float,GOP_fluency[1:]))
    GOP_integrity = list(map(float,GOP_integrity[1:]))
    IDs = IDs[1:]

    GOP_pron = list(GOP_pron[i]/10 for i in range(len(GOP_pron)))
    GOP_fluency = list(GOP_fluency[i]/10 for i in range(len(GOP_fluency)))
    GOP_integrity = list(GOP_integrity[i]/10 for i in range(len(GOP_integrity)))


with open('score_sGOP.csv') as sGOP:
    sGOP_pron = []
    sGOP_fluency = []
    sGOP_integrity = []
    IDs = []
    sGOP_reader = csv.reader(sGOP)
    for row in sGOP_reader:
        sGOP_pron.append(row[1])
        sGOP_fluency.append(row[2])
        sGOP_integrity.append(row[3])
        IDs.append(row[0])

    sGOP_pron = list(map(float,sGOP_pron[1:]))
    sGOP_fluency = list(map(float,sGOP_fluency[1:]))
    sGOP_integrity = list(map(float,sGOP_integrity[1:]))
    IDs = IDs[1:]

    sGOP_pron = list(sGOP_pron[i]/10 for i in range(len(sGOP_pron)))
    sGOP_fluency = list(sGOP_fluency[i]/10 for i in range(len(sGOP_fluency)))
    sGOP_integrity = list(sGOP_integrity[i]/10 for i in range(len(sGOP_integrity)))

with open('score_TAGOP.csv') as TAGOP:
    TAGOP_pron = []
    TAGOP_fluency = []
    TAGOP_integrity = []
    IDs = []
    TAGOP_reader = csv.reader(TAGOP)
    for row in TAGOP_reader:
        TAGOP_pron.append(row[1])
        TAGOP_fluency.append(row[2])
        TAGOP_integrity.append(row[3])
        IDs.append(row[0])

    TAGOP_pron = list(map(float,TAGOP_pron[1:]))
    TAGOP_fluency = list(map(float,TAGOP_fluency[1:]))
    TAGOP_integrity = list(map(float,TAGOP_integrity[1:]))
    IDs = IDs[1:]

    TAGOP_pron = list(TAGOP_pron[i]/10 for i in range(len(TAGOP_pron)))
    TAGOP_fluency = list(TAGOP_fluency[i]/10 for i in range(len(TAGOP_fluency)))
    TAGOP_integrity = list(TAGOP_integrity[i]/10 for i in range(len(TAGOP_integrity)))
    
with open('score_wGOP_linear.csv') as wGOP:
    wGOP_pron = []
    wGOP_fluency = []
    wGOP_integrity = []
    IDs = []
    wGOP_reader = csv.reader(wGOP)
    for row in wGOP_reader:
        wGOP_pron.append(row[1])
        wGOP_fluency.append(row[2])
        wGOP_integrity.append(row[3])
        IDs.append(row[0])

    wGOP_pron = list(map(float,wGOP_pron[1:]))
    wGOP_fluency = list(map(float,wGOP_fluency[1:]))
    wGOP_integrity = list(map(float,wGOP_integrity[1:]))
    IDs = IDs[1:]

    wGOP_pron = list(wGOP_pron[i]/10 for i in range(len(wGOP_pron)))
    wGOP_fluency = list(wGOP_fluency[i]/10 for i in range(len(wGOP_fluency)))
    wGOP_integrity = list(wGOP_integrity[i]/10 for i in range(len(wGOP_integrity)))

##### Delete



'''
with open('compare.csv') as compare:
    compare_pron_hu = []
    compare_fluency_hu = []
    compare_integrity_hu = []
    compare_pron_cu = []
    compare_fluency_cu = []
    compare_integrity_cu = []
    IDs_c = []
    compare_reader = csv.reader(compare)
    for row in compare_reader:
        compare_pron_cu.append(row[4])
        compare_fluency_cu.append(row[6])
        compare_integrity_cu.append(row[7])

        compare_pron_hu.append(row[1])
        compare_fluency_hu.append(row[2])
        compare_integrity_hu.append(row[3])

        IDs_c.append(row[0])

    compare_pron_hu = list(map(float,compare_pron_hu[1:]))
    compare_fluency_hu = list(map(float,compare_fluency_hu[1:]))
    compare_integrity_hu = list(map(float,compare_integrity_hu[1:]))

    compare_pron_hu = list(int(round(compare_pron_hu[i]/10)) for i in range(len(compare_pron_hu)))
    compare_fluency_hu = list(int(round(compare_fluency_hu[i]/10)) for i in range(len(compare_fluency_hu)))
    compare_integrity_hu = list(int(round(compare_integrity_hu[i]/10)) for i in range(len(compare_integrity_hu)))

    compare_pron_cu = list(map(float,compare_pron_cu[1:]))
    compare_fluency_cu = list(map(float,compare_fluency_cu[1:]))
    compare_integrity_cu = list(map(float,compare_integrity_cu[1:]))

    compare_pron_cu = list(int(round(compare_pron_cu[i]/10)) for i in range(len(compare_pron_cu)))
    compare_fluency_cu = list(int(round(compare_fluency_cu[i]/10)) for i in range(len(compare_fluency_cu)))
    compare_integrity_cu = list(int(round(compare_integrity_cu[i]/10)) for i in range(len(compare_integrity_cu)))

    IDs_c = IDs_c[1:]


counter = 0
for i in range(min(len(IDs_c), len(IDs_2))):
    if IDs_2[i] != IDs_c[i]:
        print (IDs_c[i] + ' : ' + IDs_2[i], end = '\t')
        print (i)
    else:
        counter += 1
print (counter)
'''
####





path = "Similarity_3.csv"
with open(path,'w') as f:
    csv_write = csv.writer(f)
    csv_head = ['method/sim', 'pearsonr', 'spearmanr', 'MIC']
    csv_write.writerow(csv_head)



### pearsonr
'''
#print(np.array(rater1_fluency).shape)
#print(np.array(ent_GOP_fluency[:len(rater1_pron)]).shape)
#print(rater1_pron)
print(rater2_pron[300:330])
print(compare_pron_hu[300:330])
print (scipy.stats.pearsonr(rater1_pron[:84] + rater1_pron[105:len(compare_pron_hu)+105-84], compare_pron_cu))
print (scipy.stats.pearsonr(compare_pron_cu, compare_pron_hu))
print (scipy.stats.pearsonr(rater2_pron, rater1_pron[:len(rater2_pron)]))
'''

pron_pearsonr_R1_R2 = scipy.stats.pearsonr(rater2_pron, rater1_pron[:len(rater2_pron)])
pron_pearsonr_R1_R3 = scipy.stats.pearsonr(rater3_pron, rater1_pron[:len(rater3_pron)])
pron_pearsonr_R2_R3 = scipy.stats.pearsonr(rater3_pron, rater2_pron[:len(rater3_pron)])
pron_pearsonr_R = (pron_pearsonr_R1_R2[0] + pron_pearsonr_R1_R3[0] + pron_pearsonr_R2_R3[0]) / 3

pron_pearsonr_fixTA_R1 = scipy.stats.pearsonr(rater1_pron, fixTAGOP_pron[:len(rater1_pron)])
pron_pearsonr_fixTA_R2 = scipy.stats.pearsonr(rater2_pron, fixTAGOP_pron[:len(rater2_pron)])
pron_pearsonr_fixTA_R3 = scipy.stats.pearsonr(rater3_pron, fixTAGOP_pron[:len(rater3_pron)])
pron_pearsonr_fixTA = (pron_pearsonr_fixTA_R1[0] + pron_pearsonr_fixTA_R2[0] + pron_pearsonr_fixTA_R3[0]) / 3

pron_pearsonr_weightedfixTA_R1 = scipy.stats.pearsonr(rater1_pron, fixTAGOP_weighted_pron[:len(rater1_pron)])
pron_pearsonr_weightedfixTA_R2 = scipy.stats.pearsonr(rater2_pron, fixTAGOP_weighted_pron[:len(rater2_pron)])
pron_pearsonr_weightedfixTA_R3 = scipy.stats.pearsonr(rater3_pron, fixTAGOP_weighted_pron[:len(rater3_pron)])
pron_pearsonr_weightedfixTA = (pron_pearsonr_weightedfixTA_R1[0] + pron_pearsonr_weightedfixTA_R2[0] + pron_pearsonr_weightedfixTA_R3[0]) / 3

pron_pearsonr_GOP_R1 = scipy.stats.pearsonr(rater1_pron, GOP_pron[:len(rater1_pron)])
pron_pearsonr_GOP_R2 = scipy.stats.pearsonr(rater2_pron, GOP_pron[:len(rater2_pron)])
pron_pearsonr_GOP_R3 = scipy.stats.pearsonr(rater3_pron, GOP_pron[:len(rater3_pron)])
pron_pearsonr_GOP = (pron_pearsonr_GOP_R1[0] + pron_pearsonr_GOP_R2[0] + pron_pearsonr_GOP_R3[0]) / 3


pron_pearsonr_wGOP_R1 = scipy.stats.pearsonr(rater1_pron, wGOP_pron[:len(rater1_pron)])
pron_pearsonr_wGOP_R2 = scipy.stats.pearsonr(rater2_pron, wGOP_pron[:len(rater2_pron)])
pron_pearsonr_wGOP_R3 = scipy.stats.pearsonr(rater3_pron, wGOP_pron[:len(rater3_pron)])
pron_pearsonr_wGOP = (pron_pearsonr_wGOP_R1[0] + pron_pearsonr_wGOP_R2[0] + pron_pearsonr_wGOP_R3[0]) / 3

pron_pearsonr_sGOP_R1 = scipy.stats.pearsonr(rater1_pron, sGOP_pron[:len(rater1_pron)])
pron_pearsonr_sGOP_R2 = scipy.stats.pearsonr(rater2_pron, sGOP_pron[:len(rater2_pron)])
pron_pearsonr_sGOP_R3 = scipy.stats.pearsonr(rater3_pron, sGOP_pron[:len(rater3_pron)])
pron_pearsonr_sGOP = (pron_pearsonr_sGOP_R1[0] + pron_pearsonr_sGOP_R2[0] + pron_pearsonr_sGOP_R3[0]) / 3

pron_pearsonr_TAGOP_R1 = scipy.stats.pearsonr(rater1_pron, TAGOP_pron[:len(rater1_pron)])
pron_pearsonr_TAGOP_R2 = scipy.stats.pearsonr(rater2_pron, TAGOP_pron[:len(rater2_pron)])
pron_pearsonr_TAGOP_R3 = scipy.stats.pearsonr(rater3_pron, TAGOP_pron[:len(rater3_pron)])
pron_pearsonr_TAGOP = (pron_pearsonr_TAGOP_R1[0] + pron_pearsonr_TAGOP_R2[0] + pron_pearsonr_TAGOP_R3[0]) / 3


fluency_pearsonr_fixTA_R1 = scipy.stats.pearsonr(rater1_fluency, fixTAGOP_fluency[:len(rater1_pron)])
fluency_pearsonr_fixTA_R2 = scipy.stats.pearsonr(rater2_fluency, fixTAGOP_fluency[:len(rater2_pron)])
fluency_pearsonr_R1_R2 = scipy.stats.pearsonr(rater2_fluency, rater1_fluency[:len(rater2_pron)])

integrity_pearsonr_fixTA_R1 = scipy.stats.pearsonr(rater1_integrity, fixTAGOP_integrity[:len(rater1_pron)])
integrity_pearsonr_fixTA_R2 = scipy.stats.pearsonr(rater2_integrity, fixTAGOP_integrity[:len(rater2_pron)])
integrity_pearsonr_R1_R2 = scipy.stats.pearsonr(rater2_integrity, rater1_integrity[:len(rater2_pron)])




'''
with open(path,'a+') as f:
    csv_write = csv.writer(f)
    data_row1 = ['pearsonr_Ent_R1', str(pron_pearsonr_ent_R1), str(fluency_pearsonr_ent_R1), str(integrity_pearsonr_ent_R1), 'pearsonr_Ent_R3', str(pron_pearsonr_ent_R3)]
    data_row2 = ['pearsonr_Ent_R2', str(pron_pearsonr_ent_R2), str(fluency_pearsonr_ent_R2), str(integrity_pearsonr_ent_R2)]
    data_row3 = ['pearsonr_R1_R2', str(pron_pearsonr_R1_R2), str(fluency_pearsonr_R1_R2), str(integrity_pearsonr_R1_R2)]
    data_row4 = ['pearsonr_GOP_R1', str(pron_pearsonr_GOP_R1), 'pearsonr_GOP_R2', str(pron_pearsonr_GOP_R2), 'pearsonr_GOP_R3', str(pron_pearsonr_GOP_R3)]
    data_row5 = ['pearsonr_wGOP_R1', str(pron_pearsonr_wGOP_R1), 'pearsonr_wGOP_R2', str(pron_pearsonr_wGOP_R2), 'pearsonr_wGOP_R3', str(pron_pearsonr_wGOP_R3)]
    data_row6 = ['pearsonr_sGOP_R1', str(pron_pearsonr_sGOP_R1), 'pearsonr_sGOP_R2', str(pron_pearsonr_sGOP_R2), 'pearsonr_sGOP_R3', str(pron_pearsonr_sGOP_R3)]
    data_row7 = ['pearsonr_TAGOP_R1', str(pron_pearsonr_TAGOP_R1), 'pearsonr_TAGOP_R2', str(pron_pearsonr_TAGOP_R2), 'pearsonr_TAGOP_R3', str(pron_pearsonr_TAGOP_R3)]
    csv_write.writerow(data_row1)
    csv_write.writerow(data_row2)
    csv_write.writerow(data_row3)
    csv_write.writerow(data_row4)
    csv_write.writerow(data_row5)
    csv_write.writerow(data_row6)
    csv_write.writerow(data_row7)
'''
### spearmanr
pron_spearmanr_R1_R2 = scipy.stats.spearmanr(rater2_pron, rater1_pron[:len(rater2_pron)])
pron_spearmanr_R1_R3 = scipy.stats.spearmanr(rater3_pron, rater1_pron[:len(rater3_pron)])
pron_spearmanr_R2_R3 = scipy.stats.spearmanr(rater3_pron, rater2_pron[:len(rater3_pron)])
pron_spearmanr_R = (pron_spearmanr_R1_R2[0] + pron_spearmanr_R1_R3[0] + pron_spearmanr_R2_R3[0]) / 3

pron_spearmanr_fixTA_R1 = scipy.stats.spearmanr(rater1_pron, fixTAGOP_pron[:len(rater1_pron)])
pron_spearmanr_fixTA_R2 = scipy.stats.spearmanr(rater2_pron, fixTAGOP_pron[:len(rater2_pron)])
pron_spearmanr_fixTA_R3 = scipy.stats.spearmanr(rater3_pron, fixTAGOP_pron[:len(rater3_pron)])
pron_spearmanr_fixTA = (pron_spearmanr_fixTA_R1[0] + pron_spearmanr_fixTA_R2[0] + pron_spearmanr_fixTA_R3[0]) / 3

pron_spearmanr_weightedfixTA_R1 = scipy.stats.spearmanr(rater1_pron, fixTAGOP_weighted_pron[:len(rater1_pron)])
pron_spearmanr_weightedfixTA_R2 = scipy.stats.spearmanr(rater2_pron, fixTAGOP_weighted_pron[:len(rater2_pron)])
pron_spearmanr_weightedfixTA_R3 = scipy.stats.spearmanr(rater3_pron, fixTAGOP_weighted_pron[:len(rater3_pron)])
pron_spearmanr_weightedfixTA = (pron_spearmanr_weightedfixTA_R1[0] + pron_spearmanr_weightedfixTA_R2[0] + pron_spearmanr_weightedfixTA_R3[0]) / 3

pron_spearmanr_GOP_R1 = scipy.stats.spearmanr(rater1_pron, GOP_pron[:len(rater1_pron)])
pron_spearmanr_GOP_R2 = scipy.stats.spearmanr(rater2_pron, GOP_pron[:len(rater2_pron)])
pron_spearmanr_GOP_R3 = scipy.stats.spearmanr(rater3_pron, GOP_pron[:len(rater3_pron)])
pron_spearmanr_GOP = (pron_spearmanr_GOP_R1[0] + pron_spearmanr_GOP_R2[0] + pron_spearmanr_GOP_R3[0]) / 3

pron_spearmanr_wGOP_R1 = scipy.stats.spearmanr(rater1_pron, wGOP_pron[:len(rater1_pron)])
pron_spearmanr_wGOP_R2 = scipy.stats.spearmanr(rater2_pron, wGOP_pron[:len(rater2_pron)])
pron_spearmanr_wGOP_R3 = scipy.stats.spearmanr(rater3_pron, wGOP_pron[:len(rater3_pron)])
pron_spearmanr_wGOP = (pron_spearmanr_wGOP_R1[0] + pron_spearmanr_wGOP_R2[0] + pron_spearmanr_wGOP_R3[0]) / 3

pron_spearmanr_sGOP_R1 = scipy.stats.spearmanr(rater1_pron, sGOP_pron[:len(rater1_pron)])
pron_spearmanr_sGOP_R2 = scipy.stats.spearmanr(rater2_pron, sGOP_pron[:len(rater2_pron)])
pron_spearmanr_sGOP_R3 = scipy.stats.spearmanr(rater3_pron, sGOP_pron[:len(rater3_pron)])
pron_spearmanr_sGOP = (pron_spearmanr_sGOP_R1[0] + pron_spearmanr_sGOP_R2[0] + pron_spearmanr_sGOP_R3[0]) / 3

pron_spearmanr_TAGOP_R1 = scipy.stats.spearmanr(rater1_pron, TAGOP_pron[:len(rater1_pron)])
pron_spearmanr_TAGOP_R2 = scipy.stats.spearmanr(rater2_pron, TAGOP_pron[:len(rater2_pron)])
pron_spearmanr_TAGOP_R3 = scipy.stats.spearmanr(rater3_pron, TAGOP_pron[:len(rater3_pron)])
pron_spearmanr_TAGOP = (pron_spearmanr_TAGOP_R1[0] + pron_spearmanr_TAGOP_R2[0] + pron_spearmanr_TAGOP_R3[0]) / 3

fluency_spearmanr_fixTA_R1 = scipy.stats.spearmanr(rater1_fluency, fixTAGOP_fluency[:len(rater1_pron)])
fluency_spearmanr_fixTA_R2 = scipy.stats.spearmanr(rater2_fluency, fixTAGOP_fluency[:len(rater2_pron)])
fluency_spearmanr_R1_R2 = scipy.stats.spearmanr(rater2_fluency, rater1_fluency[:len(rater2_pron)])

integrity_spearmanr_fixTA_R1 = scipy.stats.spearmanr(rater1_integrity, fixTAGOP_integrity[:len(rater1_pron)])
integrity_spearmanr_fixTA_R2 = scipy.stats.spearmanr(rater2_integrity, fixTAGOP_integrity[:len(rater2_pron)])
integrity_spearmanr_R1_R2 = scipy.stats.spearmanr(rater2_integrity, rater1_integrity[:len(rater2_pron)])




### MIC
def Calc_mic(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()
pron_MIC_R1_R2 = Calc_mic(rater2_pron, rater1_pron[:len(rater2_pron)])
pron_MIC_R1_R3 = Calc_mic(rater3_pron, rater1_pron[:len(rater3_pron)])
pron_MIC_R2_R3 = Calc_mic(rater3_pron, rater2_pron[:len(rater3_pron)])
pron_MIC_R = (pron_MIC_R1_R2 + pron_MIC_R1_R3 + pron_MIC_R2_R3) / 3

pron_MIC_fixTA_R1 = Calc_mic(rater1_pron, fixTAGOP_pron[:len(rater1_pron)])
pron_MIC_fixTA_R2 = Calc_mic(rater2_pron, fixTAGOP_pron[:len(rater2_pron)])
pron_MIC_fixTA_R3 = Calc_mic(rater3_pron, fixTAGOP_pron[:len(rater3_pron)])
pron_MIC_fixTA = (pron_MIC_fixTA_R1 + pron_MIC_fixTA_R2 + pron_MIC_fixTA_R3) / 3

pron_MIC_weightedfixTA_R1 = Calc_mic(rater1_pron, fixTAGOP_weighted_pron[:len(rater1_pron)])
pron_MIC_weightedfixTA_R2 = Calc_mic(rater2_pron, fixTAGOP_weighted_pron[:len(rater2_pron)])
pron_MIC_weightedfixTA_R3 = Calc_mic(rater3_pron, fixTAGOP_weighted_pron[:len(rater3_pron)])
pron_MIC_weightedfixTA = (pron_MIC_weightedfixTA_R1 + pron_MIC_weightedfixTA_R2 + pron_MIC_weightedfixTA_R3) / 3

pron_MIC_GOP_R1 = Calc_mic(rater1_pron, GOP_pron[:len(rater1_pron)])
pron_MIC_GOP_R2 = Calc_mic(rater2_pron, GOP_pron[:len(rater2_pron)])
pron_MIC_GOP_R3 = Calc_mic(rater3_pron, GOP_pron[:len(rater3_pron)])
pron_MIC_GOP = (pron_MIC_GOP_R1 + pron_MIC_GOP_R2 + pron_MIC_GOP_R3) / 3

pron_MIC_wGOP_R1 = Calc_mic(rater1_pron, wGOP_pron[:len(rater1_pron)])
pron_MIC_wGOP_R2 = Calc_mic(rater2_pron, wGOP_pron[:len(rater2_pron)])
pron_MIC_wGOP_R3 = Calc_mic(rater3_pron, wGOP_pron[:len(rater3_pron)])
pron_MIC_wGOP = (pron_MIC_wGOP_R1 + pron_MIC_wGOP_R2 + pron_MIC_wGOP_R3) / 3

pron_MIC_sGOP_R1 = Calc_mic(rater1_pron, sGOP_pron[:len(rater1_pron)])
pron_MIC_sGOP_R2 = Calc_mic(rater2_pron, sGOP_pron[:len(rater2_pron)])
pron_MIC_sGOP_R3 = Calc_mic(rater3_pron, sGOP_pron[:len(rater3_pron)])
pron_MIC_sGOP = (pron_MIC_sGOP_R1 + pron_MIC_sGOP_R2 + pron_MIC_sGOP_R3) / 3

pron_MIC_TAGOP_R1 = Calc_mic(rater1_pron, TAGOP_pron[:len(rater1_pron)])
pron_MIC_TAGOP_R2 = Calc_mic(rater2_pron, TAGOP_pron[:len(rater2_pron)])
pron_MIC_TAGOP_R3 = Calc_mic(rater3_pron, TAGOP_pron[:len(rater3_pron)])
pron_MIC_TAGOP = (pron_MIC_TAGOP_R1 + pron_MIC_TAGOP_R2 + pron_MIC_TAGOP_R3) / 3

fluency_MIC_fixTA_R1 = Calc_mic(rater1_fluency, fixTAGOP_fluency[:len(rater1_pron)])
fluency_MIC_fixTA_R2 = Calc_mic(rater2_fluency, fixTAGOP_fluency[:len(rater2_pron)])
fluency_MIC_R1_R2 = Calc_mic(rater2_fluency, rater1_fluency[:len(rater2_pron)])

integrity_MIC_fixTA_R1 = Calc_mic(rater1_integrity, fixTAGOP_integrity[:len(rater1_pron)])
integrity_MIC_fixTA_R2 = Calc_mic(rater2_integrity, fixTAGOP_integrity[:len(rater2_pron)])
integrity_MIC_R1_R2 = Calc_mic(rater2_integrity, rater1_integrity[:len(rater2_pron)])


'''
with open(path,'a+') as f:
    csv_write = csv.writer(f)
    data_row1 = ['spearmanr_Ent_R1', str(pron_spearmanr_ent_R1), str(fluency_spearmanr_ent_R1), str(integrity_spearmanr_ent_R1), 'spearmanr_Ent_R3', str(pron_spearmanr_ent_R3)]
    data_row2 = ['spearmanr_Ent_R2', str(pron_spearmanr_ent_R2), str(fluency_spearmanr_ent_R2), str(integrity_spearmanr_ent_R2)]
    data_row3 = ['spearmanr_R1_R2', str(pron_spearmanr_R1_R2), str(fluency_spearmanr_R1_R2), str(integrity_spearmanr_R1_R2)]
    data_row4 = ['spearmanr_GOP_R1', str(pron_spearmanr_GOP_R1), 'spearmanr_GOP_R2', str(pron_spearmanr_GOP_R2), 'spearmanr_GOP_R3', str(pron_spearmanr_GOP_R3)]
    data_row5 = ['spearmanr_wGOP_R1', str(pron_spearmanr_wGOP_R1), 'spearmanr_wGOP_R2', str(pron_spearmanr_wGOP_R2), 'spearmanr_wGOP_R3', str(pron_spearmanr_wGOP_R3)]
    data_row6 = ['spearmanr_sGOP_R1', str(pron_spearmanr_sGOP_R1), 'spearmanr_sGOP_R2', str(pron_spearmanr_sGOP_R2), 'spearmanr_sGOP_R3', str(pron_spearmanr_sGOP_R3)]
    data_row7 = ['spearmanr_TAGOP_R1', str(pron_spearmanr_TAGOP_R1), 'spearmanr_TAGOP_R2', str(pron_spearmanr_TAGOP_R2), 'spearmanr_TAGOP_R3', str(pron_spearmanr_TAGOP_R3)]
    csv_write.writerow(data_row1)
    csv_write.writerow(data_row2)
    csv_write.writerow(data_row3)
    csv_write.writerow(data_row4)
    csv_write.writerow(data_row5)
    csv_write.writerow(data_row6)
    csv_write.writerow(data_row7)
'''

with open(path,'a+') as f:
    csv_write = csv.writer(f)
    data_row1 = ['GOP', str(pron_pearsonr_GOP), str(pron_spearmanr_GOP), str(pron_MIC_GOP)]
    data_row2 = ['wGOP', str(pron_pearsonr_wGOP), str(pron_spearmanr_wGOP), str(pron_MIC_wGOP)]
    data_row3 = ['sGOP', str(pron_pearsonr_sGOP), str(pron_spearmanr_sGOP), str(pron_MIC_sGOP)]
    data_row4 = ['FixTA_GOP', str(pron_pearsonr_fixTA), str(pron_spearmanr_fixTA), str(pron_MIC_fixTA)]
    data_row5 = ['weighted_FixTA_GOP', str(pron_pearsonr_weightedfixTA), str(pron_spearmanr_weightedfixTA), str(pron_MIC_weightedfixTA)]
    data_row6 = ['TA_GOP', str(pron_pearsonr_TAGOP), str(pron_spearmanr_TAGOP), str(pron_MIC_TAGOP)]
    data_row7 = ['Rater', str(pron_pearsonr_R), str(pron_spearmanr_R), str(pron_MIC_R)]
    csv_write.writerow(data_row1)
    csv_write.writerow(data_row2)
    csv_write.writerow(data_row3)
    csv_write.writerow(data_row4)
    csv_write.writerow(data_row5)
    csv_write.writerow(data_row6)
    csv_write.writerow(data_row7)


'''
### MIC

Rater1 = []
Rater1.append(rater1_pron)
Rater1.append(rater1_fluency)
Rater1.append(rater1_integrity)
Rater1 = np.array(Rater1).reshape(3,len(rater1_pron))

Rater2 = []
Rater2.append(rater2_pron[:len(rater1_pron)])
Rater2.append(rater2_fluency[:len(rater1_pron)])
Rater2.append(rater2_integrity[:len(rater1_pron)])
Rater2 = np.array(Rater2).reshape(3,len(rater1_pron))

Ent_GOP = []
Ent_GOP.append(ent_GOP_pron[:len(rater1_pron)])
Ent_GOP.append(ent_GOP_fluency[:len(rater1_pron)])
Ent_GOP.append(ent_GOP_integrity[:len(rater1_pron)])
Ent_GOP = np.array(Ent_GOP).reshape(3,len(rater1_pron))

MIC_R1_R2_mic, MIC_R1_R2_tic = minepy.cstats(Rater2, Rater1, est="mic_approx")
MIC_Ent_R1_mic, MIC_Ent_R1_tic = minepy.cstats(Ent_GOP, Rater1, est="mic_approx")
MIC_Ent_R2_mic, MIC_Ent_R2_tic = minepy.cstats(Ent_GOP, Rater2, est="mic_approx")
print(MIC_R1_R2_mic)
print(MIC_Ent_R1_mic)
print(MIC_Ent_R2_mic)


#mine = MINE(alpha=0.6, c=15, est="mic_approx")

#pron_MIC_ent_R1 = mine.compute_score(rater1_pron, ent_GOP_pron[:len(rater1_pron)])
#pron_MIC_ent_R2 = mine.compute_score(rater2_pron, ent_GOP_pron[:len(rater2_pron)])
#pron_MIC_R1_R2 = mine.compute_score(rater2_pron[:len(rater1_pron)], rater1_pron)


#fluency_MIC_ent_R1 = mine.compute_score(rater1_fluency, ent_GOP_fluency[:len(rater1_pron)])
#fluency_MIC_ent_R2 = mine.compute_score(rater2_fluency, ent_GOP_fluency[:len(rater2_pron)])
#fluency_MIC_R1_R2 = mine.compute_score(rater2_fluency[:len(rater1_pron)], rater1_fluency)

#integrity_MIC_ent_R1 = mine.compute_score(rater1_integrity, ent_GOP_integrity[:len(rater1_pron)])
#integrity_MIC_ent_R2 = mine.compute_score(rater2_integrity, ent_GOP_integrity[:len(rater2_pron)])
#integrity_MIC_R1_R2 = mine.compute_score(rater2_integrity[:len(rater1_pron)], rater1_integrity)


#with open(path,'a+') as f:
#    csv_write = csv.writer(f)
#    data_row1 = ['MIC_Ent_R1', str(pron_MIC_ent_R1), str(fluency_MIC_ent_R1), str(integrity_MIC_ent_R1)]
#    data_row2 = ['MIC_Ent_R2', str(pron_MIC_ent_R2), str(fluency_MIC_ent_R2), str(integrity_MIC_ent_R2)]
 #   data_row3 = ['MIC_R1_R2', str(pron_MIC_R1_R2), str(fluency_MIC_R1_R2), str(integrity_MIC_R1_R2)]
#    csv_write.writerow(data_row1)
 #   csv_write.writerow(data_row2)
 #   csv_write.writerow(data_row3)

'''
