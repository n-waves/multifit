\toprule
RNN type & Vocabluary Size & Tokenization     & Language & MLDoc Accuracy\\
\midrule
LSTM 3   & 60k             & moses            & DE       &   94.74      \\
LSTM 4   & 30k             & sentence piece   & DE       &   95.40      \\
QRNN 4   & 60k             & moses            & DE       &   95.28     \\
QRNN 4   & 15k             & sentence piece   & DE       &   96.10      \\
\midrule
LSTM 4   & 30k             & sentence piece   & RU       &   87.27    \\
LSTM 4   & 15k             & sentence piece   & RU       &   86.47    \\
QRNN 4   & 60k             & moses            & RU       &   87.60    \\
QRNN 4   & 25k             & sentence piece   & RU       &   87.20    \\
QRNN 4   & 15k             & sentence piece   & RU       &   87.17    \\
