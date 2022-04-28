# Aspect-category Sentiment Polarity Detection

For more details, please see: [ACP: A Deep Learning Approach for Aspect-category Sentiment Polarity Detection](https://link.springer.com/chapter/10.1007/978-981-15-9516-5_14)

## Model explanation

The following is the overview of the whole repo structure, we have not included some files in this repo (due to copyrights) that are needed to run this project. But have provided the download instructions so that you can download and place the files in their respective directory.

```bash
├── data
│   ├── category_weights # this directory contains trained category_weights for different embeddings
│   │   ├── category_weights_bert.csv
│   │   ├── category_weights_elmo.csv
│   │   ├── category_weights_glove.csv
│   │   └── category_weights_word2vec.csv 
│   └── SemEval2014 # this directory contains SemEval2014 dataset
│       ├── Restaurants_Train_v2.xml
│       └── Restaurants_Test_Gold.xml
├── Embeddings # this directory contains embeddings 
│   ├── glove.6B # directory for glove
│   │   └── glove.6B.300d.txt.word2vec
│   └── Word2Vec # directory for word2vec
│       └── GoogleNews-vectors-negative300.bin.gz
├── ACP.ipynb # Main function script
├── Word2Vec_GloVe_Embeddings.py # embeddings generation model (Word2Vec, GloVe)
├── BERT_Embeddings.py # embeddings generation model (BERT)
├── ELMO_Embeddings.py # embeddings generation model (ELMo)
├── GraphPlot.py # graph generation code
├── dataset.py # for loading the XML data into dataframe
├── models.py # model architecture file
├── preprocess.py # for generating the embeddings matrix
├── install_graphviz.sh # bash script to install graphviz
├── requirements.txt # The requirements for reproducing our results
├── README.md # This instruction you are reading now
└── LICENSE # license file
```
## Requirements
* Our code is tested under Python 3.6.8
* To install all the required packages run the following command:
```shell
pip install -r requirements.txt
```

## Perparation
### Step1: Download the pre-trained embedding vectors
* Create the *Embeddings* directories
```shell
mkdir -p Embeddings/glove.6B
mkdir -p Embeddings/Word2Vec
```
* #### For GloVe Embedding
Download and unzip GloVe vectors(`glove.6B.zip`) from https://nlp.stanford.edu/projects/glove/ or using the script below:
```shell
cd Embeddings
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove.6B
rm glove.6B.zip
```

* #### For Word2Vec Embedding
```shell
cd Embeddings/Word2Vec
```
Download the word2vec vectors from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

### Step2: Download the Dataset

* We used [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) dataset. The training dataset can be downloaded from [here](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/) and testing dataset can be downloaded from [here](http://metashare.elda.org/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/)
```shell
mkdir -p data/SemEval2014
```

## Training and Testing
Run: `ACP.ipynb` 

## Reference

If you found this code useful, please consider citing the following paper:

Kumar, A., Dahiya, V., Sharan, A. (2021). **ACP: A Deep Learning Approach for Aspect-category Sentiment Polarity Detection.** In: *Machine Intelligence and Soft Computing. Advances in Intelligent Systems and Computing, vol 1280.* Springer, Singapore. https://doi.org/10.1007/978-981-15-9516-5_14


```bibtex
@inproceedings{kumar2021acp,
    title="ACP: A Deep Learning Approach for Aspect-category Sentiment Polarity Detection",
    author="Kumar, Ashish and Dahiya, Vasundhra and Sharan, Aditi",
    editor="Bhattacharyya, Debnath and Thirupathi Rao, N.",
    booktitle="Machine Intelligence and Soft Computing",
    year="2021",
    publisher="Springer Singapore",
    address="Singapore",
    pages="163--176",
    isbn="978-981-15-9516-5"
}
```
