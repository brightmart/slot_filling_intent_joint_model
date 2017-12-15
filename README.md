Joint model for intent detection and slot filling based on attention, input alignment and knowledge

Introduction:
-------------------------------------------------------------------------------------
1.intent detection and slot filling joint model which share encoding information

2.incorporate knowledge information with embedding for both intent detection and slot filling. this embedding share the same embedding space with slots output.

3.use bi-direction RNN and CNN to do intent detection

4.use slots middle output as a feature for intent detection to boost performance

5.toy task: input a sequence of natural number, such as [5,7,2,6,8,3].
for slot filling: count label each number to 0 or 1. if sum of a number together with its previous and next number is great than a threshold(such as 14), we mark it as 1. otherwise 0.
in this case, output of slot filling will be:[0,0,1,1,1,0]
for intent detection, count how many number totally is marked as 1. in this case, output of intent will be:3.

Performance:
-------------------------------------------------------------------------------------
|-----------|---intent_tmall--|---intent_tmall(similiarity)---|---intent_alime---|---intent_alime(similiarity)---|

|---Performance---|---95.37%---|---72.0%---|---93.0%---|---62.9%---|


Usage:
-------------------------------------------------------------------------------------
1.train the model: train() of a1_joint_intent_slots_model.py

2.test the model: predict() of a1_joint_intent_slots_model.py

![alt text](https://github.com/brightmart/slot_filling_intent_joint_model/blob/master/JOINT_MODEL.JPG)


Reference:
-------------------------------------------------------------------------------------
1.Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling,

https://arxiv.org/pdf/1609.01454.pdf

2.阿里AI Labs王刚解读9小时卖出百万台的“天猫精灵” | 高山大学（GASA）,

http://www.sohu.com/a/206109679_473283

3.史上最全！阿里智能人机交互的核心技术解析 
https://yq.aliyun.com/articles/277907?spm=5176.100244.teamhomeleft.54.SKEyCU
