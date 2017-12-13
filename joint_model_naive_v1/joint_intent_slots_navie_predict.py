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
from joint_intent_slots_naive_model import joint_naive_model
from a1_data_util import *
import math
import pickle

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.99, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",15,"max sentence length") #100
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_step", 1000, "how many step to validate.") #1000做一次检验
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_integer("intent_num_classes",31,"number of classes for intent")
tf.app.flags.DEFINE_integer("vocab_size",382,"vocabulary size for input(x)")
tf.app.flags.DEFINE_integer("slots_num_classes",4,"number of classes for slots")

#create session and load the model from checkpoint
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
FLAGS.batch_size = 1
sequence_length_batch = [FLAGS.sequence_length] * FLAGS.batch_size
model = joint_naive_model(FLAGS.intent_num_classes, FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate,
                          FLAGS.sequence_length, FLAGS.vocab_size, FLAGS.embed_size, FLAGS.hidden_size,
                          sequence_length_batch, FLAGS.slots_num_classes, FLAGS.is_training)
# initialize Saver
saver = tf.train.Saver()
print('restoring Variables from Checkpoint!')
saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
# load vocabulary for intent and slot name
word2id = create_or_load_vocabulary(None)
id2word = {value: key for key, value in word2id.items()}
word2id_intent = create_or_load_vocabulary_intent(None)
id2word_intent = {value: key for key, value in word2id_intent.items()}
word2id_slotname = create_or_load_vocabulary_slotname_save(None)
id2word_slotname = {value: key for key, value in word2id_slotname.items()}

def main(_):
    sentence=u'开灯' #u'帮我打开厕所的灯'
    #indices=[240, 277, 104, 274, 344, 259, 19, 372, 235, 338, 338, 338, 338, 338, 338] #[283, 180, 362, 277, 99, 338, 338, 338, 338, 338, 338, 338, 338, 338, 338] #u'帮我把客厅的灯打开'
    intent, slots=predict(sentence)
    print(sentence)
    print("intent:%s" %intent)
    for slot_name,slot_value in slots.items():
        print('slot_name:{},slot_value:{}'.format(slot_name, slot_value))

    predict_interactive()

def predict(sentence):
    """
    :param sentence: a sentence.
    :return: intent and slots
    """
    sentence_indices=index_sentence_with_vocabulary(sentence,word2id,FLAGS.sequence_length)
    feed_dict = {model.x: np.reshape(sentence_indices,(1,FLAGS.sequence_length))}
    logits_intent,logits_slots = sess.run([model.logits_intent,model.logits_slots], feed_dict)
    intent,slots=get_result(logits_intent,logits_slots,sentence_indices)
    return intent,slots

def predict_interactive():
    sys.stdout.write("Please Input Story.>")
    sys.stdout.flush()
    question = sys.stdin.readline()
    while question:
        #1.predict using quesiton
        intent, slots=predict(question)
        #2.print
        print("intent:%s" % intent)
        for slot_name, slot_value in slots.items():
            print('slot_name:{}-->slot_value:{}'.format(slot_name, slot_value))
        #3.read new input
        print("Please Input Story>")
        sys.stdout.flush()
        question = sys.stdin.readline()



def get_result(logits_intent,logits_slots,sentence_indices):
    index_intent= np.argmax(logits_intent[0]) #index of intent
    intent=id2word_intent[index_intent]

    slots=[]
    indices_slots=np.argmax(logits_slots[0],axis=1) #[sequence_length]
    for i,index in enumerate(indices_slots):
        slots.append(id2word_slotname[index])
    slots_dict={}
    for i,slot in enumerate(slots):
        word=id2word[sentence_indices[i]]
        #print(i,"slot:",slot,";word:",word,";slots!=O:",slots!=O)
        if slot!=O and word!=PAD and word!=UNK:
            slots_dict[slot]=word
    return intent,slots_dict

if __name__ == "__main__":
    tf.app.run()