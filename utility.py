import os
from zipfile import ZipFile

import numpy as np


"""
    Auto Extract Models.zip if folder Models is not exist
"""


def auto_extract_if_needed():
    if os.path.isdir("Models"):
        return

    with ZipFile("Models.zip", "r") as f:
        f.extractall()


def fastaToNgram(fastaSequenceFile, ngram):
    f = open(fastaSequenceFile, "r")
    lines = f.readlines()
    f.close()
    fasttext_input_sequence_dic = {}
    #    threshold=float(lines[0][:-1])
    fastaSequence = ""
    temp = lines[0]
    temp = temp.replace(">sp|", "").replace(">", "")
    proteinID = temp[:temp.find("|")]
    for line in lines:
        if line.find(">") < 0:
            if line[-1] == "\n":
                fastaSequence += line[:-1]
            else:
                fastaSequence += line
        else:
            fasttext_input_sequence_dic[proteinID] = fastaSequence
            temp = line
            temp = temp.replace(">sp|", "").replace(">", "")
            proteinID = temp[:temp.find("|")]
            fastaSequence = ""
    fasttext_input_sequence_dic[proteinID] = fastaSequence
    # cho nay ko gan ngram=3 vi voi bai tnf, feature tot nhat la combine ngram2 va ngram3
    for key in fasttext_input_sequence_dic.keys():
        fastaSequence = fasttext_input_sequence_dic.get(key)
        fasttext_input_sequence = ""
        i = 0
        while i < len(fastaSequence) - ngram + 1:
            for j in range(ngram):
                fasttext_input_sequence += fastaSequence[i + j]
            fasttext_input_sequence += " "
            i = i + 1
        #        print(fasttext_input_sequence)
        fasttext_input_sequence_dic[key] = fasttext_input_sequence[:-1]

    return fasttext_input_sequence_dic


def create_word_vector(embedding_file):
    word_int={}
    f=open(embedding_file,"r")
    lines=f.readlines()
    for line in lines[1:]:
        word_int[line.split()[0]]=[float(line.split()[1])] #dim=1
    del word_int["</s>"]
    return word_int


def init(word_int):
    input_word_int={}
    for key in word_int.keys():
        input_word_int[key]=[0,0]#dim=2
    return input_word_int


def count_word(aList, word):
    count=0
    for l in aList:
        if l==word:
            count+=1
    return count


def create_svm_input_from_dict(
        embedding_file,
        fasttext_input_sequence_dic
):
    word_int = create_word_vector(embedding_file)
    input_word_int = init(word_int)
    features = []
    for fasttext_input_sequence in fasttext_input_sequence_dic.values():
        for ngram in fasttext_input_sequence.split():
            count = count_word(fasttext_input_sequence[:-1].split(),ngram)
            if word_int.get(ngram) is not None:
                input_word_int[ngram] = [count*word_int.get(ngram)[0]]#dim1
        for key in input_word_int.keys():
            features.append(float('{:.3f}'.format(input_word_int.get(key)[0])))

    return features


def create_svm_input_from_one_seq(
        embedding_file1,
        embedding_file2,
        fasttext_input_sequence1,
        fasttext_input_sequence2,
):
    features_1 = create_svm_input_from_dict(
        embedding_file=embedding_file1,
        fasttext_input_sequence_dic={"seq": fasttext_input_sequence1}
    )

    features_2 = create_svm_input_from_dict(
        embedding_file=embedding_file2,
        fasttext_input_sequence_dic={"seq": fasttext_input_sequence2}
    )

    return features_1 + features_2


def labelToOneHot(label):# 0--> [1 0], 1 --> [0 1]
    label = label.reshape(len(label), 1)
    label = np.append(label, label, axis = 1)
    label[:,0] = label[:,0] == 0;
    return label
