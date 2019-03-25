# Web Requests Clustering via AutoEncoder and K-Means

### Main algorithm
1. Preprocess text.
2. Make text embedding via autoencoder.
3. Predict class using pre-trained K-Means clustering model.

### Preprocessing
1. Make vocabulary of chars used in sequences.
2. Process input vectors and map them to sequences of ints: `["how", "are", "you"] -> [[30, 14, 65], [3, 80, 5], [45, 10, 35]`.
3. Pad all sequences with zeros or truncate them to make them equal-length.
4. Insert `<START>` and `<END>` tokens.

### Training
1. Autoencoder is usual Encoder-Decoder architecture that learns to reconstruct the given sequence. It consists of Embedding layer and two LSTM layers.
2. During clustering, only encoder part is used. Encoder outputs are input text embeddings that are used to classify these texts. These embeddings are proven to be better than raw texts and TF-IDF features because they better reflect semantics of texts.
3. Text embeddings are then processed by K-Means algorithm with n_clusters=25. Final weights are saved in `models/AE_kmeans_65k.joblib`. 


### Example reconstructions (`autoencoder.predict()`)

```
**vector**: Mozilla/5.0 (Linux; Android 5.1; ZTE T630 Build/LMY47D) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 YaBrowser/17.3.1.383.00 Mobile Safari/537.36
**reconstruction**: M o z i l l a / 5 . 0   ( L i n u x ;   A n d r o i d   4 . 4 . 2 ;   i n d o n   B u l l ;   1 1 6 

**vector**: 1508bb9b84c5b5d8b03b413fad8e4c19b85e7ad3ea04a59f533e6dee97c649d16456bc0b0401a177363fd9c73e0f3d9f\"\"\
**reconstruction**: 1 5 0 1 a d 8 b 0 0 4 c a 5 0 4 e 5 0 4 1 1 1 8 e 5 0 4 1 1 1 9 7 e 4 1 1 1 9 7 1 4 1 1 2 4 1 4 

**vector**: 14' and if(((/*!SelEcT*/ conv(substring(hex(cast(TABLE_NAME as char CHARACTER SET utf8)),29,2),16,10) fRom information_schema.TABLES where TABLE_SCHEMA=0x753335323830395f73686f706d6d OrDer by TABLE_NAME limit 35,1)& 16)>0,1,0) and 3>'1 
**reconstruction**: 1 4 '   a n d   i f ( ( ( / * ! S e l E c T * /   c o n v ( s u b s t r i n g ( h e x ( c a s t ( c 

**vector**: ÐÐ¾Ð¿Ð¾Ð»Ð½Ð¸ÑÐµÐ»ÑÐ½Ð¾ Ð²Ñ Ð¼Ð¾Ð¶ÐµÑÐµ Ð¿ÑÐ¸Ð¾Ð±ÑÐµÑÑÐ¸ <a href="/category/tickets/football/la-liga/5702/tour-real-valladolid/"> ÑÑÑ Ð½Ð° Ð¼Ð°ÑÑ Ð ÐµÐ°Ð» ÐÐ°Ð´ÑÐ¸Ð´ - ÐÐ°Ð»ÑÑÐ´Ð¾Ð»Ð¸Ð´</a>. ÐÑÐ¾Ð¶Ð¸Ð²Ð°Ð½Ð¸Ðµ Ð² ÐÐ°Ð´ÑÐ¸Ð´Ðµ. 3 Ð½Ð¾ÑÐ¸ / 4 Ð´Ð½Ñ.
**reconstruction**: Ð ¾ Ð ½ Ð ¾ Ð ½ Ð ¾ Ð ² Ð °   Ð ° Ð ½ Ð ¾ Ð ² Ð ° Ñ Ð ¸ Ð ½ Ð °   Ð ¾ Ð ² Ð ° Ñ Ð ¸ Ð ½ Ð ° Ñ Ð ¸ Ð

**vector**: <BLANK> 
**reconstruction**: < B L A N K >
```

### Project structure
* `pipeline.ipynb` is main file with both training and prediction
* `download_and_predict.py` is equivalent Python script I used for parallel processing
* `vectors/sample-requests.csv` contains 122 sample requests you can use for testing prediction and training


### Requirements
* Python 3.6
* You can install all necessary packages via 
`pip install -r requirements.txt`