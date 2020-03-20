# Game-of-Thrones-Dataset
Game of Thrones Dataset for narrative structure extraction through scene linking

There is a csv file that have a 444 total number of scenes with 16 columns to characterize scenes. 
You can use the utility.py file to do the following:
1. Generate one hot encoding of the important datas such as speaking characters, appearing characters, keywords, entity mentions.
2. Generate TF-IDF reprenetation of the important datas such as speaking characters, appearing characters, keywords, entity mùentions.
3. Generate document embeddings using pretrained doc2vec model of the scene texts
4. Generate sentence embedding of using BERT transformer model for the texts of each scene
