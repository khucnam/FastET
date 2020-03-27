#FastET
This is the public site for the paper under revison named: "Boosting performance of electron transport protein prediction model using different word embedding types"

 
Living organisms receive necessary energy substances directly from cellular respiration. The completion of electron storage and transportation requires the process of cellular respiration with the aid of electron transport chains. Therefore, the work of deciphering electron transport proteins is inevitably needed. In order to identify proteins, classification performance has a prompt dependence on the choice of methods for feature extraction and machine learning algorithm. In this study, protein sequences are treated as natural language sentences comprising words. The nominated word embedding-based feature sets, hinged on the word embedding modulation and protein motif frequencies, were useful for feature choosing. Five word embedding types and a variety of conjoint multiple features was examined for such feature selection. The support vector machine algorithm consequentially was employed to identify electron transport proteins. The statistics of models within the 5-fold cross validation including average accuracy, specificity, sensitivity as well as MCC rates are 98.46%, 99.36%, 95.26%, and 0.955, respectively. Such metrics in the independent test are 96.82%, 97.16%, 95.76%, and 0.9, respectively. Compared to state-of-the-art predictors, the prososed method can generate more preferable performance above all metrics. These figures indicated the proposed classification model effectiveness with the task of determining electron transport proteins. Furthermore, this study replenishes a basis for futuristic research which enables the enrichment of natural language processing tactics in bioinformatics research.



LIBRARY REQUIREMENTS
	We will need to install some basic packages to run the programs as followed:
		git version 2.15.1.windows.2
		python 3.6.5
		numpy 1.14.3
		pandas 0.23.0
		sklearn 0.20.2
		
		
INSTRUCTION:

1. Using git bash to clone all the required files in "YOUR FOLDER" folder
git clone https://github.com/khucnam/FastET

2. Perform the prediction using the following command:
	
	python Predict.py your_fasta_file.fasta

	("your_fasta_file.fasta" file contains the sequences you want to classify. Please see the "sample.fasta" as an example.)

3. Open the corresponding XXX_Result.csv file to see the results.
   XXX can be one of the following NoSub-CBOW, NoSubFN, NoSubSK, SubCBOW, SubSK. These are 5 different word embedding feature type used. 
   In each result file, there are 2 columms: first one contains the protein ID, the next column contains the probability of the protein sequence to be electron transport proteins.

