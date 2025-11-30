# 概述
本项目以kaggle时序比赛[store-sales-time-series-forecasting竞赛](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)为背景，采用了LSTM和Transformer两种方法去预测未来16天的序列。我们没有进行复杂的特征工程，只是进行了简单的插值和处理，并取得了 0.47513(LSTM) 和 0.46949（Transformer）。

可以看出，Transformer的提升不大，这是因为Transformer更复杂，需要的训练数据更大、对特征工程要求更大，如果您感兴趣，可以自行修改。