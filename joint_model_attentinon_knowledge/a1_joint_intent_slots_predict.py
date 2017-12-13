# -*- coding: utf-8 -*-
#prediction using model.process--->1.load data. 2.create session. 3.feed data. 4.predict
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
import os
import codecs
from a1_seq2seq_attention_model import seq2seq_attention_model
from data_util import load_test_data,load_vocab_as_dict,_GO,_PAD,_EOS,_UNK
from tflearn.data_utils import  pad_sequences
from a1_preprocess import preprocess_english_file

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 400, "Batch size for training/evaluating.") #400 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","ckpt_ai_challenger_translation/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",30,"max sentence length")
tf.app.flags.DEFINE_integer("decoder_sent_length",30,"length of decoder inputs")
tf.app.flags.DEFINE_integer("embed_size",256,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_integer("hidden_size",256,"hidden size")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")
tf.app.flags.DEFINE_string("predict_target_file","ckpt_ai_challenger_translation/s2q_attention.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("data_en_test_path",'./data/test_a_20170923.sgm',"target file path for final prediction")
tf.app.flags.DEFINE_string("data_en_test_processed_path",'./data/test_a_20170923.sgm.processed',"target file path for final prediction")
tf.app.flags.DEFINE_string("vocabulary_cn_path","./data/vocabulary.zh","path of traning data.")
tf.app.flags.DEFINE_string("vocabulary_en_path","./data/vocabulary.en","path of traning data.")
tf.app.flags.DEFINE_boolean("use_beam_search",True,"whether use beam search during decoding.")

def main(_):
    #1.load test data
    vocab_cn, vocab_en = load_vocab_as_dict(FLAGS.vocabulary_cn_path, FLAGS.vocabulary_en_path)
    flag_data_en_test_processed_path=os.path.exists(FLAGS.data_en_test_processed_path)
    print("processed of english source file exists or not:",flag_data_en_test_processed_path)
    if not flag_data_en_test_processed_path:
        preprocess_english_file(FLAGS.data_en_test_path, FLAGS.data_en_test_processed_path)
    test=load_test_data(FLAGS.data_en_test_processed_path, vocab_en, FLAGS.decoder_sent_length)
    print("test[0:10]:",test[0:10])
    test = pad_sequences(test, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
    sequence_length_batch = [FLAGS.sequence_length] * FLAGS.batch_size

    #2.create session,model,feed data to make a prediction
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = seq2seq_attention_model(len(vocab_cn), FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                        FLAGS.decay_rate, FLAGS.sequence_length, len(vocab_en), FLAGS.embed_size,
                                        FLAGS.hidden_size, sequence_length_batch,FLAGS.is_training,decoder_sent_length=FLAGS.decoder_sent_length,
                                        l2_lambda=FLAGS.l2_lambda,use_beam_search=FLAGS.use_beam_search)
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        #feed data, to get logits
        number_of_test_data = len(test)
        print("number_of_test_data:", number_of_test_data)
        index = 0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        decoder_input=np.array([[vocab_cn[_GO]] + [vocab_cn[_PAD]] * (FLAGS.decoder_sent_length - 1)]*FLAGS.batch_size)
        print("decoder_input:", decoder_input.shape)
        decoder_input = np.reshape(decoder_input, [-1, FLAGS.decoder_sent_length])
        print("decoder_input:",decoder_input.shape)
        vocab_cn_index2word = dict([val, key] for key, val in vocab_cn.items())

        for start, end in zip(range(0, number_of_test_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_test_data + 1, FLAGS.batch_size)):
            predictions = sess.run(model.predictions, # predictions:[batch_size,decoder_sent_length]
                                           feed_dict={model.input_x: test[start:end],
                                                      model.decoder_input: decoder_input,
                                                      model.dropout_keep_prob: 1})  # 'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            output_sentence_list= get_label_using_logits(predictions, vocab_cn_index2word, vocab_cn)
            # 7. write question id and labels to file system.
            for sentence in output_sentence_list:
                predict_target_file_f.write(sentence+"\n")
        predict_target_file_f.close()

def get_label_using_logits(predictions, vocab_cn_index2word, vocab_cn):
    """
    :param predictions: array as [batch_size,decoder_sent_length]
    :param vocab_cn_index2word:
    :param vocab_cn:
    :return:
    """
    #print("predictions:",predictions.shape)
    eos_index=vocab_cn[_EOS]
    result_list=[]
    for ii,selected_token_ids in enumerate(predictions):
        #print("selected_token_ids0:",selected_token_ids.shape)
        selected_token_ids=list(selected_token_ids)
        #print("selected_token_ids1:",selected_token_ids)
        if eos_index in selected_token_ids:
            eos_index = selected_token_ids.index(eos_index)
            selected_token_ids=selected_token_ids[0:eos_index]
        output_sentence = "".join([vocab_cn_index2word[index] for index in selected_token_ids])
        if _PAD in output_sentence:
            output_sentence=output_sentence[0:output_sentence.find(_PAD)]
        result_list.append(output_sentence)
    return result_list
if __name__ == "__main__":
    tf.app.run()