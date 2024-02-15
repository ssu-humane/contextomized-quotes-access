# A Data-Centric Contrastive Embedding Framework for Contextomized News Quotes Detection (QuoteCSE++)

## Task: Contextomized Quote Detection
Let a given news article be *X:(H, B)*, where *H* is the news headline, and *B* is the body text. Our task is to predict a binary label *Y* indicating whether the headline quote in *H* is either modified or contextomized by referring to the body-text quotes.
![problem](https://github.com/ssu-humane/data-centric-contextomized-quote-framework/assets/80903024/75bbb382-87ea-4050-acc9-dc489567c35a)

## Method: QuoteCSE++

We present QuoteCSE++, a data-centric contrastive learning framework. 

This figure illustrates the QuoteCSE++ framework.
![framework](https://github.com/ssu-humane/data-centric-contextomized-quote-framework/assets/80903024/0a926133-0151-4246-bf34-1eb785fcf9f3)

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
|"이대론 그리스처럼 파탄"(A debt crisis, like Greece, is on the horizon)|건강할 때 재정을 지키지 못하면 그리스처럼 될 수도 있다"(If we do not maintain our fiscal health, we may end up like Greece) <br/> "강력한 ‘지출 구조조정’을 통해 허투루 쓰이는 예산을 아껴 필요한 곳에 투입해야 한다"(Wasted budgets should be reallocated to areas in need through the reconstruction of public expenditure)|Contextomized <br/> (1)|
|"불필요한 모임 일절 자제"(Avoid unnecessary gatherings altogether)|"저도 백신을 맞고 해서 여름에 어디 여행이라도 한번 갈 계획을 했었는데..."(Since being vaccinated, I had planned to travel somewhere in the summer, but...) <br/> "어떤 행위는 금지하고 어떤 행위는 허용한다는 개념이 아니라 불필요한 모임과 약속, 외출을 일제 자제하고…."(It is not a matter of prohibiting or permitting specific activities, but of avoiding unnecessary gatherings, appointments, and going out altogether...)|Modified <br/> (0)|


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
