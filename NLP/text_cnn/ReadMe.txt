1. data/dataall_train.txt 训练数据 4600W
   data/dataall_test.txt 测试数据 250W
   data/Vocabulary/vocab_processor 词表

2. output/model 模型保存的位置
   output/result 输出的关键词的保存位置

3. utils/ 为数据、结果等的处理脚本

4. src/prepareDate.py 输入数据预处理的脚本
   src/TextCNN.py TextCNN+attention网络结构的脚本
   src/cnn_attention.py 模型训练的脚本

5. python cnn_attention.py 开始训练模型
