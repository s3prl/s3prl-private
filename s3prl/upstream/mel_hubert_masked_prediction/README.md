#  Mel-HuBERT pre-training on Original HuBERT objective but with hard label instead 
## Basic information
This is the implementation of HuBERT which takes fbank feature as input. And its pre-training objective is to predict the hard clustering label of fbank feature on masked time steps.
## Notes 
- When testing on downstream, you need to implement your own feature extractor in order to convert the input waveform to fbank feature. Make sure the feature extractor could produce same feature as you used in pre-training phase. Specifically, you should change 107,108 line in s3prl/s3prl/upstream/mel_hubert_masked_prediction/builder.py. (You have to implement your own process_input_data function and preprocessor.)