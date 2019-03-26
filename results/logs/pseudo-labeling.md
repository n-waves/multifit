## 100 ex. pseudo labeling bootstrapping


## Laser pseudo labeling bootstrapping
```
Processing data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/models/sp15k/qrnn_nl4.m
Generating pseduolabels data/mldoc/de-1-laser-en1-ps
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/de-1-laser-en1/de.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁.', '▁,', '▁der', '▁die', 'en', '▁und', 's', '▁in', 'er', "▁&'"]
/home/pczapla/anaconda3/envs/fastaiv1/lib/python3.7/site-packages/torch/utils/cpp_extension.py:152: UserWarning: 

                               !! WARNING !!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Your compiler (c++) may be ABI-incompatible with PyTorch!
Please use a compiler that is ABI-compatible with GCC 4.9 and above.
See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.

See https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6
for instructions on how to install GCC 4.9 or higher.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                              !! WARNING !!

  warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
Generating train dataset of size 1000, the accuracy is 0.997
   0                                                  1  preds
0  3  Tokio (Reuter) - Der Dollar ist am Donnerstag ...      3
1  3  Kairo (Reuter) - Die ägyptische Zentralbank se...      3
2  2  Bonn (Reuter) - Wegen einer Bombendrohung ist ...      2
3  0  Berlin (Reuter) - Die Bahn AG will mit Hilfe p...      0
4  3  08.15 Uhr MEZ - Deutsche Aktien nach den Rekor...      3
Generating dev dataset of size 1000, the accuracy is 0.91
   0                                                  1  preds
0  1  New York (Reuter) - Das Vertrauen der US-Verbr...      1
1  2  Tokio (Reuter) - Russische Patrouillenboote ha...      2
2  2  Paris (Reuter) - Bei der Volksabstimmung in Al...      2
3  2  Belgrad (Reuter) - Die serbische Polizei hat n...      2
4  0  München (Reuter) - Der Stuttgarter Bosch-Konze...      0
Processing data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/models/sp15k/qrnn_nl4.m
Generating pseduolabels data/mldoc/es-1-laser-en1-ps
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/es-1-laser-en1/es.dev.csv
Data lm, trn: 13013, val: 1445
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', '▁la', '▁.', 's', '▁en', '▁el', '▁y', '▁a', '▁que']
Generating train dataset of size 1000, the accuracy is 0.988
   0                                                  1  preds
0  3  LONDRES, 5 sep (Reuter) - El dólar se mantenía...      3
1  2  MADRID, 30 dic (Reuter) - La Generalitat de Va...      2
2  3  PARIS, 30 jun (Reuter) - La Bolsa de París neg...      3
3  0  MADRID, 23 dic (Reuter) - La agencia de valore...      0
4  0  MADRID, 4 Feb (Reuter) - El Banco Bilbao Vizca...      0
Generating dev dataset of size 1000, the accuracy is 0.879
   0                                                  1  preds
0  0  NUEVA YORK, 11 abr (Reuter) - MCI Communicatio...      0
1  3  FRANCFORT, 17 jun (Reuter) - La Bolsa de Franc...      3
2  2  BONN, 3 jun (Reuter) - Un destacado miembro de...      1
3  2  LONDRES, 3 sep (Reuter) - El secretario de Def...      2
4  3  MADRID, 3 oct (Reuter) - Las acciones de Pryca...      3
Processing data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/models/sp15k/qrnn_nl4.m
Generating pseduolabels data/mldoc/fr-1-laser-en1-ps
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/fr-1-laser-en1/fr.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁de', '▁,', 's', '▁.', "'", '▁la', '▁le', '▁et', '▁l', '▁à']
Generating train dataset of size 1000, the accuracy is 0.993
   0                                                  1  preds
0  2  WASHINGTON, 13 septembre, Reuter - Les Etats-U...      2
1  1  PARIS, 10 juillet, Reuter - L'audit des financ...      1
2  2  MOSCOU, 29 mai, Reuter - Après l'accord interv...      2
3  2  PARIS, 1er octobre, Reuter - Le groupe communi...      2
4  0  LONDRES, 3 juin, Reuter - National Grid Group ...      0
Generating dev dataset of size 1000, the accuracy is 0.887
   0                                                  1  preds
0  1  PARIS, 30 décembre, Reuter - Zodiac . Chiffre ...      0
1  0  AJACCIO, 11 décembre, Reuter - Une charge de 7...      2
2  0  BRUXELLES, 26 décembre, Reuter - 1997 s'annonc...      0
3  0  PARIS, 26 septembre, Reuter - Alcatel Alsthom ...      0
4  1  NEW YORK, 25 octobre, Reuter - La hausse plus ...      1
Processing data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/models/sp15k/qrnn_nl4.m
Generating pseduolabels data/mldoc/it-1-laser-en1-ps
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/it-1-laser-en1/it.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁di', '▁e', "▁&'", "'", '▁il', '▁la', '▁in', 'e']
Generating train dataset of size 1000, the accuracy is 0.987
   0                                                  1  preds
0  3  MILANO, 6 nov (Reuter) - La lira recupera ai p...      3
1  1  MILANO, 20 giugno (Reuter) - Lo stacco dividen...      1
2  3  MILANO, 20 set (Reuter) - Olivetti entra nel t...      3
3  1  LONDRA, 2 aprile (Reuter) - L'aggregato moneta...      1
4  3  Oro Londra fix ore 10,30 - 4 nov - $378,65. (c...      3
Generating dev dataset of size 1000, the accuracy is 0.819
   0                                                  1  preds
0  0  L'istituto prevede un aumento dell'utile opera...      0
1  1  FRANCOFORTE, 18 dic (Reuter) - La Bundesbank a...      1
2  1  TOKIO, 28 agosto (Reuter) - Il ministro delle ...      1
3  1  ROMA, 23 luglio (Reuter) - Il presidente del C...      1
4  1  MONACO, 19 marzo (Reuter) - Il ministro delle ...      1
Processing data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/models/sp15k/qrnn_nl4.m
Generating pseduolabels data/mldoc/ru-1-laser-en1-ps
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/ru-1-laser-en1/ru.dev.csv
Data lm, trn: 9195, val: 1021
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁.', '▁в', 'а', 'и', 'е', '▁и', 'й', '▁на', 'х']
Generating train dataset of size 1000, the accuracy is 0.996
   0                                                  1  preds
0  0  КИЕВ, 20 июн (Рейтер) - Нацбанк Украины планир...      0
1  3  МИНСК, 13 фев (Рейтер) - Курс белорусского руб...      3
2  0  САНКТ-ПЕТЕРБУРГ, 25 авг (Рейтер) - Астробанк (...      0
3  0  MOSCOW, Feb 7 (Reuter) - U.S. plane-maker Boei...      0
4  2  В данном обзоре казахстанской прессы приводитс...      2
Generating dev dataset of size 1000, the accuracy is 0.837
   0                                                  1  preds
0  0  ТБИЛИСИ, 25 мар (Рейтер) - Партнерский Фонд, с...      0
1  3  МОСКВА, 3 ноя (Рейтер) - Казахстанская Межбанк...      3
2  1  КИЕВ, 25 июл (Рейтер) - Нацбанк Украины рассмо...      1
3  0  МОСКВА, 2 дек (Рейтер) - АО Уралсвязьинформ пр...      0
4  2  В данном обзоре киргизской прессы приводится к...      2
Processing data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m
Max vocab: 15000
Cache dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k
Model dir: /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/models/sp15k/qrnn_nl4.m
Generating pseduolabels data/mldoc/zh-1-laser-en1-ps
Loading validation /home/pczapla/workspace/ulmfit-multilingual/data/mldoc/zh-1-laser-en1/zh.dev.csv
Data lm, trn: 13500, val: 1500
Data cls, trn: 1000, val: 1000
Data tst, trn: 1000, val: 4000
Size of vocabulary: 15000
First 20 words in vocab: ['xxunk', 'xxpad', 'xxbos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep', '<unk>', '▁', '▁,', '▁的', '▁。', '▁年', '▁、', '▁在', '▁一', '▁是', '▁中', '▁有']
Generating train dataset of size 1000, the accuracy is 0.992
   0                                                  1  preds
0  1  〔路透社紐約10日電〕　　芝加哥聯邦準備銀行總裁墨斯克週四表示,他預期1997年國內生產總值...      1
1  0  〔路透社台北14日電〕台灣合作金庫週四將2週､1個月及2個月內的附條件交易利率全開在5.20...      0
2  1  〔路透社倫敦6日電〕　　在英國工黨政府賦予央行利率自主權後,英國央行在其新的首次貨幣政策委員...      1
3  3  〔路透社東京4日電〕　　東京股市週一收盤下跌,但在短暫跌破關鍵支撐19,500點後縮減跌幅....      3
4  2  美國總統克林頓接受明報訪問時表示,美國是貫徹始終地支持中英''聯合聲明''作為香港未來的基石...      2
Generating dev dataset of size 1000, the accuracy is 0.817
   0                                                  1  preds
0  0  〔路透社台北20日電〕　　台灣塑膠類週一早盤上漲,經紀商表示,主要是因為近期原物料價格上漲及...      0
1  2  〔路透社華盛頓2日電〕比利時央行總裁弗沛雷茲週三表示,義大利里拉被低估,但美元可望攀升.  ...      1
2  0  〔路透社吉隆坡29日電〕　　吉隆坡股市周二收市微升.分析師指二線股有散戶吸納,助長市場升勢,...      3
3  2  〔路透社香港26日電〕　　香港明報周四報導,面對台灣當局的"務實外交",和"台獨"傾向,中國...      2
4  0  [路透社上海6日電]    據上海証券報周五報導,有關專家就滬市四家上市公司法人股通過拍賣進...      0
Python 3.7.0 (default, Oct  9 2018, 10:31:47) 
Type 'copyright', 'credits' or 'license' for more information
```