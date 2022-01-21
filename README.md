# Listen-Attend-Spell

This code implements the [Listen, Attend, Spell](https://arxiv.org/abs/1508.01211) paper using PyTorch which transcribes speech to text. The results of the speller on test audio data can be seen in submission_lm.csv. The attentions folder contains attention maps for some randomly chosen training batches.

The architecture implemented is an LSTM based Seq2Seq Encoder-Decoder with Self-Attention. The dataset used is audio mel spectrogram data (~5 GB). The model is trained using AWS EC2 instances.
