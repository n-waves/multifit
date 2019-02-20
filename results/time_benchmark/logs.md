# Results

## Set-up. 
 - Num Tokens 15K
 - GPU V100
 - LM BPTT = 70
 - LM BS = 64
 - CLAS BS = 32

| Model          |   LSTM    |   QRNN    |
|----------------|-----------|-----------|
| LM ms/batch    |   143ms   |   71ms    |
| CLAS ms/batch  |   467ms   |   156ms   |


```
> python results/time_benchmark/qrnn_benchmark.py

Vocab size 14513                                                                                                                                                                                                    
QRNN
LM
epoch     train_loss  valid_loss  accuracy
1         6.326089                                                                                                                                                                                                                        
Total time: 00:11
Batch size torch.Size([64, 70])
Params = 22 MM
Training time is 71.0 ms per batch
CLAS
epoch     train_loss  valid_loss  accuracy
1         0.712603                                                                                                                                                                                                                      
Total time: 00:10
Batch size torch.Size([32, 1445])
Params = 22 MM
Training time is 156.0 ms per batch
LSTM
LM
epoch     train_loss  valid_loss  accuracy
1         6.262911                                                                                                                                                                                                                        
Total time: 00:21
Batch size torch.Size([64, 70])
Params = 37 MM
Training time is 143.0 ms per batch
CLAS
epoch     train_loss  valid_loss  accuracy
1         0.706715                                                                                                                                                                                                                      
Total time: 00:32
Batch size torch.Size([32, 1445])
Params = 37 MM
Training time is 467.0 ms per batch
```