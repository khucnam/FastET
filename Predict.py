import numpy as np
import pickle
from sklearn.externals import joblib
import sys

from utility import \
    fastaToNgram,\
    create_word_vector, \
    init, count_word, \
    create_svm_input_from_dict, \
    create_svm_input_from_one_seq, \
    labelToOneHot,\
    auto_extract_if_needed


def run(model_file, x_test):
    x_test = np.array(x_test)
    try:
        classifier=joblib.load(model_file)
    except (IOError, pickle.UnpicklingError, AssertionError):
        print(pickle.UnpicklingError)
        return True

    y_pred = classifier.predict_proba(x_test)
    return y_pred[0][1] #all the second values


def predict(
        model_name,
        output_file_path,
        fasttext_input_sequence_dic1,
        fasttext_input_sequence_dic2,
        embedding_file1_path,
        embedding_file2_path,
):
    f = open("{}.{}".format(model_name, output_file_path), "w")
    f.write("ProteinID,Probability\n")
    for proteinID in fasttext_input_sequence_dic1.keys():
        f.write(proteinID + ",")
        fasttext_input_sequence1 = fasttext_input_sequence_dic1.get(proteinID)
        fasttext_input_sequence2 = fasttext_input_sequence_dic2.get(proteinID)
        for cla in ["A"]:
            model_file = "Models/{}/{}.pickle_model.pkl".format(model_name, cla)
            features = create_svm_input_from_one_seq(
                embedding_file1_path,
                embedding_file2_path,
                fasttext_input_sequence1,
                fasttext_input_sequence2,
            )
            answerForOneClass = run(model_file=model_file, x_test=[features])
            f.write(str("{:.3f}".format(answerForOneClass)) + ",")
        f.write("\n")
    f.close()


if __name__ == "__main__":

    auto_extract_if_needed()

    inputFile = sys.argv[1]
    outputFile = "Result.csv" #w: allow overwrite
    print("input file ", inputFile)

    #ngram2
    fasttext_input_sequence_dic1 = fastaToNgram(inputFile, 2)

    #ngram3
    fasttext_input_sequence_dic2 = fastaToNgram(inputFile, 3)

    predict(
        model_name="NoSub-CBOW",
        output_file_path=outputFile,
        fasttext_input_sequence_dic1=fasttext_input_sequence_dic1,
        fasttext_input_sequence_dic2=fasttext_input_sequence_dic2,
        embedding_file1_path="fastText embedding vectors/fastText embedding vectors/NoSub-CBOW/ngram1.GENSIM.embedding.train.A.vec",
        embedding_file2_path="fastText embedding vectors/fastText embedding vectors/NoSub-CBOW/ngram3.GENSIM.embedding.train.A.vec",
    )

    predict(
        model_name="NoSubFN",
        output_file_path=outputFile,
        fasttext_input_sequence_dic1=fasttext_input_sequence_dic1,
        fasttext_input_sequence_dic2=fasttext_input_sequence_dic2,
        embedding_file1_path="fastText embedding vectors/fastText embedding vectors/NoSubFN/ngram1.keras.embedding.epoch100.vec",
        embedding_file2_path="fastText embedding vectors/fastText embedding vectors/NoSubFN/ngram3.keras.embedding.epoch100.vec",
    )

    predict(
        model_name="NoSubSK",
        output_file_path=outputFile,
        fasttext_input_sequence_dic1=fasttext_input_sequence_dic1,
        fasttext_input_sequence_dic2=fasttext_input_sequence_dic2,
        embedding_file1_path="fastText embedding vectors/fastText embedding vectors/NoSubSK/ngram1.GENSIM.embedding.train.A.vec",
        embedding_file2_path="fastText embedding vectors/fastText embedding vectors/NoSubSK/ngram3.GENSIM.embedding.train.A.vec",
    )

    predict(
        model_name="SubCBOW",
        output_file_path=outputFile,
        fasttext_input_sequence_dic1=fasttext_input_sequence_dic1,
        fasttext_input_sequence_dic2=fasttext_input_sequence_dic2,
        embedding_file1_path="fastText embedding vectors/fastText embedding vectors/SubCBOW/ngram1.dfSubword.embedding.train.A.vec",
        embedding_file2_path="fastText embedding vectors/fastText embedding vectors/SubCBOW/ngram3.dfSubword.embedding.train.A.vec"
    )

    predict(
        model_name="SubSK",
        output_file_path=outputFile,
        fasttext_input_sequence_dic1=fasttext_input_sequence_dic1,
        fasttext_input_sequence_dic2=fasttext_input_sequence_dic2,
        embedding_file1_path="fastText embedding vectors/fastText embedding vectors/SubSK/ngram1.dfSubword.embedding.train.A.vec",
        embedding_file2_path="fastText embedding vectors/fastText embedding vectors/SubSK/ngram3.dfSubword.embedding.train.A.vec",
    )

    print("Thank you for using FastET!!! Please check the prediction results in 5 different XXX_Result.csv files corresponding to 5 word embedding-based feature types")



