# slot_filling_intent_joint_model
attention based joint model for intent detection and slot filling 

包含：
1）共享encodig信息的slot intent联合模型、

2）使用mulit-hot表示的知识编码、

3）结合CNN和RNN的意图识别

4)  使用slots任务输出作为intent的一个特征，提高效果。Toy task: 5000步后loss从0.73降低到0.53

4）在toy task上的训练和测试方法。
Toy task: 输入一串数字，如：【5，7，2，6，8，3】
槽填充部分：将附近元素（元素左边、自己和右边）之和大于15的标记为1，否则标记为0；
意图识别部分(intent),找出符合条件的元素的总个数。

使用:
训练：train()
预测：predict()

Reference:
1.Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling,https://arxiv.org/pdf/1609.01454.pdf
2.阿里AI Labs王刚解读9小时卖出百万台的“天猫精灵” | 高山大学（GASA）,http://www.sohu.com/a/206109679_473283
