# -*- coding: utf-8 -*-
#prediction using model.process--->1.load data. 2.create session. 3.feed data. 4.predict
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
import os
from joint_intent_slots_knowledge_model import joint_knowledge_model
from a1_data_util import *
import math
import pickle

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.99, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint_skill3/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",25,"max sentence length") #100
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_step", 1000, "how many step to validate.") #1000做一次检验
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_boolean("enable_knowledge",True,"whether to use knwoledge or not.")
tf.app.flags.DEFINE_string("knowledge_path","knowledge_skill3","file for data source") #skill3_train_20171114.txt


#create session and load the model from checkpoint
# load vocabulary for intent and slot name
word2id = create_or_load_vocabulary(None,FLAGS.knowledge_path)
id2word = {value: key for key, value in word2id.items()}
word2id_intent = create_or_load_vocabulary_intent(None,FLAGS.knowledge_path)
id2word_intent = {value: key for key, value in word2id_intent.items()}
word2id_slotname = create_or_load_vocabulary_slotname_save(None,FLAGS.knowledge_path)
id2word_slotname = {value: key for key, value in word2id_slotname.items()}
knowledge_dict=load_knowledge(FLAGS.knowledge_path)

intent_num_classes=len(word2id_intent)
vocab_size=len(word2id)
slots_num_classes=len(id2word_slotname)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
FLAGS.batch_size = 1
sequence_length_batch = [FLAGS.sequence_length] * FLAGS.batch_size
model = joint_knowledge_model(intent_num_classes, FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate,
                          FLAGS.sequence_length, vocab_size, FLAGS.embed_size, FLAGS.hidden_size,
                          sequence_length_batch, slots_num_classes, FLAGS.is_training)
# initialize Saver
saver = tf.train.Saver()
print('restoring Variables from Checkpoint!')
saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))


slot_values_file = FLAGS.knowledge_path+'/slot_values.txt'
jieba.load_userdict(slot_values_file)

def main(_):
    sentence=u'开灯' #u'帮我打开厕所的灯'
    #indices=[240, 277, 104, 274, 344, 259, 19, 372, 235, 338, 338, 338, 338, 338, 338] #[283, 180, 362, 277, 99, 338, 338, 338, 338, 338, 338, 338, 338, 338, 338] #u'帮我把客厅的灯打开'
    intent,intent_logits, slots,slot_list=predict(sentence)
    print(sentence)
    print('intent:{},intent_logits:{}'.format(intent, intent_logits))
    #for slot_name,slot_value in slots.items():
    #    print('slot_name:{},slot_value:{}'.format(slot_name, slot_value))
    for i,element in enumerate(slot_list):
        slot_name,slot_value=element
        print('slot_name:{},slot_value:{}'.format(slot_name, slot_value))

    predict_interactive()

def predict(sentence,enable_knowledge=1):
    """
    :param sentence: a sentence.
    :return: intent and slots
    """
    print("FLAGS.knowledge_path====>:",FLAGS.knowledge_path)
    sentence_indices=index_sentence_with_vocabulary(sentence,word2id,FLAGS.sequence_length,knowledge_path=FLAGS.knowledge_path)
    y_slots= get_y_slots_by_knowledge(sentence,FLAGS.sequence_length,enable_knowledge=enable_knowledge,knowledge_path=FLAGS.knowledge_path)
    print("predict.y_slots:",y_slots)
    feed_dict = {model.x: np.reshape(sentence_indices,(1,FLAGS.sequence_length)),model.y_slots:np.reshape(y_slots,(1,FLAGS.sequence_length)),model.dropout_keep_prob:1.0}
    logits_intent,logits_slots = sess.run([model.logits_intent,model.logits_slots], feed_dict)
    intent,intent_logits,slots,slot_list=get_result(logits_intent,logits_slots,sentence_indices)
    return intent,intent_logits,slots,slot_list

def get_y_slots_by_knowledge(sentence,sequence_length,enable_knowledge=1,knowledge_path=None):
    """get y_slots using dictt.e.g. dictt={'slots': {'全部范围': '全', '房间': '储藏室', '设备名': '四开开关'}, 'user': '替我把储藏室四开开关全关闭一下', 'intent': '关设备<房间><全部范围><设备名>'}"""
    #knowledge_dict=#{'储藏室': '房间', '全': '全部范围', '四开开关': '设备名'}
    user_speech_tokenized=tokenize_sentence(sentence,knowledge_path=knowledge_path) #['替', '我', '把', '储藏室', '四开', '开关', '全', '关闭', '一下']
    result=[word2id_slotname[O]]*sequence_length
    if enable_knowledge=='1' or enable_knowledge==1:
        for i,word in enumerate(user_speech_tokenized):
            slot_name=knowledge_dict.get(word,None)
            if slot_name is not None:
                try:
                    result[i]=word2id_slotname[slot_name]
                except:
                    pass
    return result

def predict_interactive():
    sys.stdout.write("Please Input Story.>")
    sys.stdout.flush()
    question = sys.stdin.readline()
    enable_knowledge=1
    while question:
        if question.find("disable_knowledge")>=0:
            enable_knowledge=0
            print("knowledge disabled")
            print("Please Input Story>")
            sys.stdout.flush()
            question = sys.stdin.readline()
        elif question.find("enable_knowledge")>=0:
            enable_knowledge=1
            #3.read new input
            print("knowledge enabled")
            print("Please Input Story>")
            sys.stdout.flush()
            question = sys.stdin.readline()

        #1.predict using quesiton
        intent, intent_logits,slots,slot_list=predict(question,enable_knowledge=enable_knowledge)
        #2.print
        print('intent:{},intent_logits:{}'.format(intent, intent_logits))
        #for slot_name, slot_value in slots.items():
        #    print('slot_name:{}-->slot_value:{}'.format(slot_name, slot_value))
        for i, element in enumerate(slot_list):
            slot_name, slot_value = element
            print('slot_name:{},slot_value:{}'.format(slot_name, slot_value))
        #3.read new input
        print("Please Input Story>")
        sys.stdout.flush()
        question = sys.stdin.readline()


def get_result(logits_intent,logits_slots,sentence_indices):
    index_intent= np.argmax(logits_intent[0]) #index of intent
    intent_logits=logits_intent[0][index_intent]
    print("intent_logits:",index_intent)
    intent=id2word_intent[index_intent]

    slots=[]
    indices_slots=np.argmax(logits_slots[0],axis=1) #[sequence_length]
    for i,index in enumerate(indices_slots):
        slots.append(id2word_slotname[index])
    slots_dict={}
    slot_list=[]
    for i,slot in enumerate(slots):
        word=id2word[sentence_indices[i]]
        print(i,"slot:",slot,";word:",word)
        if slot!=O and word!=PAD and word!=UNK:
            slots_dict[slot]=word
            slot_list.append((slot,word))
    return intent,intent_logits,slots_dict,slot_list

if __name__ == "__main__":
    tf.app.run()