# A Data-Centric Contrastive Embedding Framework for Contextomized News Quotes Detection (QuoteCSE++)

## Task: Contextomized Quote Detection
Let a given news article be *X:(H, B)*, where *H* is the news headline, and *B* is the body text. Our task is to predict a binary label *Y* indicating whether the headline quote in *H* is either modified or contextomized by referring to the body-text quotes.
![problem](https://github.com/ssu-humane/data-centric-contextomized-quote-framework/assets/80903024/b8cc9d20-41fe-436a-818e-b370908f79d7)

## Method: QuoteCSE++

We present QuoteCSE++, a data-centric contrastive learning framework. 

This figure illustrates the QuoteCSE++ framework.
![framework](https://github.com/ssu-humane/data-centric-contextomized-quote-framework/assets/80903024/81a4f864-e530-4761-982c-8a7223564d87)

We proposed the BERT-like transformer encoder to represent the semantics of news quotes. In addition, we implemented a classifier to classify whether the headline quote *H* is contextomized or not using embedding obtained by QuoteCSE++. 
QuoteCSE++ is designed based on journalism ethics to learn effective quote representations. QuoteCSE++ maximizes the semantic similarity between the headline quote and the matched quote in the body text while minimizing the similarity for other unmatched quotes in the same or other articles.

We obtain embeddings of headline quote and body quotes from QuoteCSE++. The headline quote embedding is **u**, and the **v** is the body text embedding that are concatenated two body text quotes similar to the **u**. To detect the contextomized quote, We implemented a binary MLP classifier with **u**, **v**, **|u-v|**, **u*v** as input.



### Pretraining corpus
```
data/modified_sample.pkl
data/verbatim_sample.pkl
```
We present a sampled dataset of unlabeled corpus used for the QuoteCSE++ pretraining. Each data instance consists of title quote, positive sample, and negative sample. The positive and negative samples were selected by SentenceBERT, and the assignments are updated during training.

### Contextomized quote detection
```
data/contextomized_quote.pkl
```

The dataset is also available though the HuggingFace Hub
```python
from datasets import load_dataset

data = load_dataset('humane-lab/contextomized-quote')
```
We introduce a dataset of 3,000 news articles for detecting contextomized news quotes.
- Label 1: The headline quote is *contextomized*, which refers to the excerpt of words with semantic changes from the original statement.
- Label 0: The headline quote is *modified*. It keeps the semantics of original expression but it is a different phrase or sentence.

**Examples**
|title quote|body quotes|label|
|------|---|:---:|
|"빨간옷 고집 마라"(Don't insist on wearing Red clothes)|"이번 주말부터는 시민 속으로 좀 더 파고들기 위해 빨간색에 구애받지 말고 자율적으로 복장을 하도록 했다"(Starting this weekend, we decided not to wear red clothes mandatory to encourage more citizens to join us) <br/> "집회 참가자들의 복장이 빨간색 일색이다보니 너무 튄다는 지적이 있었고 일부 시민들이 반감을 나타내는 경우도 있었다"(Since the rally participants' costumes were all red, some citizens criticized it as being too conspicuous and expressed their displeasure)|Contextomized <br/> (1)|
|"AI 두려워할 필요 없어"(Don't need to fear AI)|"우리는 인공지능을 두려워할 필요가 없다"(We don't need to fear artificial intelligence) <br/> "오히려 우리는 인공지능이 세상에 가져다줄 놀랄 만큼 많은 이점을 기대해야 한다"(Instead, we should look forward to the numerous remarkable benefits that artificial intellignece will bring to the world)|Modified <br/> (0)|


## Usage

### QuoteCSE pretraining
```python
python train.py 
```
You can obtain the pretrained QuoteCSE checkpoints [here](https://drive.google.com/drive/folders/1XL34nX27vYDiJUZyhHaWqlD8cno672mH?usp=sharing).

### QuoteCSE-based detection
```python
python contextomized_quote_detection.py 
```
