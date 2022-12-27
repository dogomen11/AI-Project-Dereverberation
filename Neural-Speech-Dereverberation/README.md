# Neural-Speech-Dereverberation
Machine and Deep Learning models for speech dereverberation

## Data
- LibriSpeech for speech audio files [1]. Available: https://www.openslr.org/12
- Omni and MARDY dataset for Room Impulse Responses (RIRs) [2, 3]. Available: http://isophonics.org/content/room-impulse-response-data-set
  and https://www.commsp.ee.ic.ac.uk/~sap/resources/mardy-multichannel-acoustic-reverberation-database-at-york-database/
- BUT Speech@FIT Reverb Database for retransmitted data [4]. Available: https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database

## Models

- MLP and LSTM with "Context Window"
- Late Reverberation Supression LSTM [5] 
- FD-NDLP (WPE + frequency domain) [6]. 
  Implementation taken from https://github.com/helianvine/fdndlp
- U-net for speech dereverberation [7]. U-net architecture is based on image segmentation, available: https://github.com/milesial/Pytorch-UNet
- Late Reverberation Supression U-net (proposed method, based on [5, 7] ideas)
- GAN training with U-net generator [7]
 

Speech Enhancement Example with U-net generator:

<img src="example.png" width="900" height="200" />


## Metrics
- Perceptual Evaluation of Speech Quality (PESQ)
- Cepstral Distorsion (CD)
- Log Likelihood Ratio (LLR)
- Frequency-Weighted Segmental Signal to Noise Ratio (fwSNRseg)
- Speech to Reverberation Modulation Energy Ratio (SRMR)

Python implementation is taken from: https://github.com/schmiph2/pysepm

## Citing
If you use code or any ideas from here, please cite our
[publication at arXiv](https://arxiv.org/abs/2110.02144)

## References

[1] Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015.

[2] R. Stewart and M. Sandler, "Database of omnidirectional and B-format room impulse responses," 2010 IEEE International Conference on Acoustics, Speech and Signal Processing, Dallas, TX, USA, 2010, pp. 165-168, doi: 10.1109/ICASSP.2010.5496083.

[3] J. Y. C. Wen, N. D Gaubitch, E. a. P. Habets, T. Myatt, and P. a. Naylor, "Evaluation of Speech Dereverberation Algorithms using the MARDY Database," Proc. Intl. Workshop Acoust. Echo Noise Control  (IWAENC)}, pp. 12-15, 2006. 

[4] I. Szöke, M. Skácel, L. Mošner, J. Paliesek and J. Černocký, ''Building and evaluation of a real room impulse response dataset'', in IEEE Journal of Selected Topics in Signal Processing, vol. 13, no. 4, pp. 863-876, Aug. 2019, doi: 10.1109/JSTSP.2019.2917582.

[5] Yan Zhao, Deliang Wang, Buye Xu y Tao Zhang, ''Late Reverberation Supression using Recurrent Neural Networks with Long Short-Term Memory''. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.

[6] T. Nakatani, T. Yoshioka, K. Kinoshita, M. Miyoshi and B. Juang, "Speech Dereverberation Based on Variance-Normalized Delayed Linear Prediction," in IEEE Transactions on Audio, Speech, and Language Processing, vol. 18, no. 7, pp. 1717-1731, Sept. 2010, doi: 10.1109/TASL.2010.2052251.

[7] Ori Ernst, Shlomo E. Chazan, Sharon Gannot and Jacob Goldberger, "Speech Dereverberation Using Fully Convolutional Networks". Faculty of Engineering, Bar-Ilan University, 3 Apr, 2019.
