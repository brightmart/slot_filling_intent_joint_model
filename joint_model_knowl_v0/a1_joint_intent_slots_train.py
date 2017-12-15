# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from a1_seq2seq_attention_model import  seq2seq_attention_model
from data_util import load_data,load_vocab_as_dict
from data_util_vocabulary import _build_vocab,_build_vocab_en
import os,math
import word2vec
import pickle

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 92, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.99, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","ckpt_ai_challenger_translation/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",30,"max sentence length") #100
tf.app.flags.DEFINE_integer("decoder_sent_length",30,"max sentence length") #100

tf.app.flags.DEFINE_integer("embed_size",1000,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",20,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_step", 6000, "how many step to validate.") #1000做一次检验
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","./data/ai_challenger_translation.bin-128","word2vec's vocabulary and vectors") #zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
tf.app.flags.DEFINE_integer("hidden_size",1000,"hidden size")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")
tf.app.flags.DEFINE_integer("vocabulary_size_en",60000,"vocabulary size of english") #100000
tf.app.flags.DEFINE_integer("vocabulary_size_cn",100000,"vocabulary size of chinese") #100000
tf.app.flags.DEFINE_string("data_folder","./data","path of traning data.")
tf.app.flags.DEFINE_string("data_cn_path","./data/train.zh","path of traning data.")
tf.app.flags.DEFINE_string("data_en_path","./data/train.en","path of traning data.")
tf.app.flags.DEFINE_string("data_en_processed_path","./data/train.en.processed","path of traning data.")
tf.app.flags.DEFINE_string("data_cn_valid_path","./data/valid.en-zh.zh.sgm","path of traning data.")
tf.app.flags.DEFINE_string("data_en_valid_path","./data/valid.en-zh.en.sgm","path of traning data.")
tf.app.flags.DEFINE_string("vocabulary_cn_path","./data/vocabulary.zh","path of traning data.")
tf.app.flags.DEFINE_string("vocabulary_en_path","./data/vocabulary.en","path of traning data.")
tf.app.flags.DEFINE_boolean("test_mode",False,"whether it is test mode. if test mode, only use training size will be only 10,000.")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #PART1############################################PREPARE DATA FOR TRAINING############################################
    #1.create vocab
    #_build_vocab(FLAGS.data_en_path, FLAGS.vocabulary_en_path, FLAGS.vocabulary_size_en)
    _build_vocab_en(FLAGS.word2vec_model_path, FLAGS.vocabulary_en_path, FLAGS.vocabulary_size_en)
    _build_vocab(FLAGS.data_cn_path, FLAGS.vocabulary_cn_path, FLAGS.vocabulary_size_cn)
    #2.load vocab
    vocab_cn, vocab_en=load_vocab_as_dict(FLAGS.vocabulary_cn_path, FLAGS.vocabulary_en_path)
    vocab_en_index2word=dict([val,key] for key,val in vocab_en.items()) #get reverse order.
    #3.load data
    train,valid=load_data(FLAGS.data_folder, FLAGS.data_cn_path, FLAGS.data_en_path,FLAGS.data_en_processed_path, vocab_cn, vocab_en,FLAGS.data_cn_valid_path,FLAGS.data_en_valid_path,FLAGS.sequence_length,test_mode=FLAGS.test_mode)
    trainX, trainY_input,trainY_output = train
    testX, testY_input,testY_output = valid
    #4. print sample data
    print("trainX:", trainX[0:10]);print("trainY_input:",trainY_input[0:10]);print("trainY_output:",trainY_output[0:10])
    print("testX:", testX[0:10]);print("testY_input:",testY_input[0:10]);print("testY_output:",testY_output[0:10])
    sequence_length_batch = [FLAGS.sequence_length] * FLAGS.batch_size
    # PART1############################################PREPARE DATA FOR TRAINING#############################################
    # PART2############################################TRAINING#############################################################
    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        model = seq2seq_attention_model(len(vocab_cn), FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                        FLAGS.decay_rate, FLAGS.sequence_length, len(vocab_en), FLAGS.embed_size,
                                        FLAGS.hidden_size, sequence_length_batch,FLAGS.is_training,decoder_sent_length=FLAGS.decoder_sent_length, l2_lambda=FLAGS.l2_lambda)
        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocab_en_index2word, model,word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch = sess.run(model.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(trainX)
        print("number_of_training_data:", number_of_training_data)
        previous_eval_loss = 10000
        best_eval_loss = 10000
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:#print sample to have a look
                    print("trainX[start:end]:", trainX[start:end])
                feed_dict = {model.input_x: trainX[start:end], model.dropout_keep_prob: 0.5}
                feed_dict[model.decoder_input] = trainY_input[start:end]
                feed_dict[model.input_y_label] = trainY_output[start:end]
                curr_loss, curr_acc, _ = sess.run([model.loss_val, model.accuracy, model.train_op],feed_dict)
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                if counter % 50 == 0:
                    print("seq2seq_with_attention==>Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %
                        (epoch, counter, math.exp(loss / float(counter)) if (loss / float(counter)) < 20 else 10000.000,acc / float(counter)))
                ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                if start % (FLAGS.validate_step * FLAGS.batch_size) == 0:
                    eval_loss, _ = do_eval(sess, model, testX, testY_input,testY_output, batch_size)
                    print("seq2seq_with_attention.validation.part. previous_eval_loss:",
                          math.exp(previous_eval_loss) if previous_eval_loss < 20 else 10000.000, ";current_eval_loss:",
                          math.exp(eval_loss) if eval_loss < 20 else 10000.000)
                    if eval_loss > previous_eval_loss:  # if loss is not decreasing
                        # reduce the learning rate by a factor of 0.5
                        print("seq2seq_with_attention==>validation.part.going to reduce the learning rate.")
                        learning_rate1 = sess.run(model.learning_rate)
                        lrr = sess.run([model.learning_rate_decay_half_op])
                        learning_rate2 = sess.run(model.learning_rate)
                        print("seq2seq_with_attention==>validation.part.learning_rate1:", learning_rate1," ;learning_rate2:", learning_rate2)
                    else:  # loss is decreasing
                        if eval_loss < best_eval_loss:
                            print("seq2seq_with_attention==>going to save the model.eval_loss:",
                                  math.exp(eval_loss) if eval_loss < 20 else 10000.000, ";best_eval_loss:",
                                  math.exp(best_eval_loss) if best_eval_loss < 20 else 10000.000)
                            # save model to checkpoint
                            save_path = FLAGS.ckpt_dir + "model.ckpt"
                            saver.save(sess, save_path, global_step=epoch)
                            best_eval_loss = eval_loss
                    previous_eval_loss = eval_loss
                    ##VALIDATION VALIDATION VALIDATION PART######################################################################################################

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        #test_loss, test_acc = do_eval(sess, model, testX, testY, batch_size)
    pass
    # PART2############################################TRAINING#############################################################
def assign_pretrained_word_embedding(sess, vocabulary_index2word, model, word2vec_model_path=None):#vocab_size
    """
    assign pretrained word embedding to parameter of the model.
    :param sess:
    :param vocabulary_index2word: a dict of vocabulary of source side,{index:word}. in our experiment, it is a dict of english vocabulary
    :param vocab_size:
    :param model:
    :param word2vec_model_path: word-embedding path
    :return: no return, value assigned and applied to the parameter of the model
    """
    print("using pre-trained word emebedding.started.word2vec_model_path:", word2vec_model_path)
    #1.load word-vector pair as dict, which was generated by word2vec
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    #2.assign vector for each word in vocabulary list; otherwise,init it.
    vocab_size=len(vocabulary_index2word)
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:#no embedding for this word-->init it randomly.
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    #3.assign(and invoke the operation) vocabulary list to variable of our model
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度
def do_eval(sess, model, evalX, evalY_input,evalY_output, batch_size):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {model.input_x: evalX[start:end], model.dropout_keep_prob: 1.0}
        feed_dict[model.decoder_input] = evalY_input[start:end]
        feed_dict[model.input_y_label] = evalY_output[start:end]
        curr_eval_loss, logits, curr_eval_acc, pred = sess.run([model.loss_val, model.logits, model.accuracy, model.predictions],feed_dict)
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)



if __name__ == "__main__":
    tf.app.run()


if __name__ == "__main__":
    tf.app.run()