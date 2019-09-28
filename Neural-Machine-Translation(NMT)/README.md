# Neural Language_Translator

The objective is to convert an English sentence to its German counterpart using a Neural Machine Translation (NMT) system. We shall use three different model to train model.

Models are;

- Simple Language Model
- Seq2seq model
- Attention model

To evaluate model, we shall use accuracy , sparse categorical accuracy and sparse_categorical_crossentropy.

# Modules

pandas==0.24.3
numpy==1.16.0
re==2.2.1
sklearn==0.20.3
tensorflow==1.12.0
keras==2.2.4

References

- RNN: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

- LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- GRU: https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be

- GRU vs LSTM : https://datascience.stackexchange.com/questions/14581/when-to-use-gru-over-lstm

- Word Embedding: https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa

- Out-of-Vocab: https://medium.com/@shabeelkandi/handling-out-of-vocabulary-words-in-natural-language-processing-based-on-context-4bbba16214d5

- Seq2seq using neural net:Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).

- Seq2seq:https://medium.com/analytics-vidhya/a-must-read-nlp-tutorial-on-neural-machine-translation-the-technique-powering-google-translate-c5c8d97d7587

- Teacher forcing: https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/

- word-level seq2seq: https://towardsdatascience.com/word-level-english-to-marathi-neural-machine-translation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7

- Eng-Fr: https://medium.com/@dev.elect.iitd/neural-machine-translation-using-word-level-seq2seq-model-47538cba8cd7

- Attention: https://medium.com/datalogue/attention-in-keras-1892773a4f22

- Attention-model: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

- Attention model wrapper: https://github.com/neonbjb/ml-notebooks/blob/master/keras-seq2seq-with-attention/keras_translate_notebook.ipynb
