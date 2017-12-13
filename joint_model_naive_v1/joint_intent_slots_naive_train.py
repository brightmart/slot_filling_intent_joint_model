# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
import os
from joint_intent_slots_naive_model import joint_naive_model
from a1_data_util import generate_training_data
import math
import pickle

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.99, "Rate of decay for learning rate.") #0.87一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",15,"max sentence length") #100
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_step", 1000, "how many step to validate.") #1000做一次检验
tf.app.flags.DEFINE_integer("hidden_size",128,"hidden size")
tf.app.flags.DEFINE_float("l2_lambda", 0.0001, "l2 regularization")

tf.app.flags.DEFINE_integer("intent_num_classes",31,"number of classes for intent")
tf.app.flags.DEFINE_integer("vocab_size",382,"vocabulary size for input(x)")
tf.app.flags.DEFINE_integer("slots_num_classes",4,"number of classes for slots")
tf.app.flags.DEFINE_string("data_source","knowledge/sht_20171125.txt","file for data source")


def main(_):
    #1. load data
    traing_data, valid_data, test_data = generate_training_data(FLAGS.data_source)
    print("training_data:",type(traing_data),"")
    x_train, y_intent_train, y_slots_train=traing_data
    x_valid, y_intent_valid, y_slots_valid = valid_data
    x_test, y_intent_test, y_slots_test = test_data

    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        sequence_length_batch=[FLAGS.sequence_length]*FLAGS.batch_size
        model=joint_naive_model(FLAGS.intent_num_classes, FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
                 FLAGS.vocab_size, FLAGS.embed_size,FLAGS.hidden_size, sequence_length_batch,FLAGS.slots_num_classes,FLAGS.is_training)
        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
        curr_epoch = sess.run(model.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(x_train)
        print("number_of_training_data:", number_of_training_data)
        previous_eval_loss = 10000
        best_eval_loss = 10000
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc_intent,acc_slot, counter = 0.0,0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),
                                  range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:#print sample to have a look
                    print("trainX[start:end]:", x_train[start:end])
                feed_dict = {model.x: x_train[start:end], model.dropout_keep_prob: 0.5}
                feed_dict[model.y_intent] = y_intent_train[start:end]
                feed_dict[model.y_slots] = y_slots_train[start:end]
                curr_loss, curr_acc_intent,curr_acc_slot, _ = sess.run([model.loss_val, model.accuracy_intent,model.accuracy_slot, model.train_op],feed_dict)
                loss, counter, acc_intent,acc_slot = loss + curr_loss, counter + 1, acc_intent + curr_acc_intent,acc_slot+curr_acc_slot
                if counter % 50 == 0:
                    print("joint_intent_slots_navie==>Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy_intent:%.3f\tTrain Accuracy_slot:%.3f" %(epoch, counter, loss / float(counter),acc_intent / float(counter),acc_slot / float(counter)))

                if start % (FLAGS.validate_step * FLAGS.batch_size) == 0:#evaluation.
                    eval_loss,acc_intent,acc_slot = do_eval(sess, model, x_valid, y_intent_valid, y_slots_valid, batch_size)
                    print("joint_intent_slots_navie.validation.part. previous_eval_loss:",previous_eval_loss ,";current_eval_loss:",eval_loss,";acc_intent:",acc_intent,";acc_slot:",acc_slot)
                    if eval_loss > previous_eval_loss:  # if loss is not decreasing
                        # reduce the learning rate by a factor of 0.5
                        print("joint_intent_slots_navie==>validation.part.going to reduce the learning rate.")
                        learning_rate1 = sess.run(model.learning_rate)
                        lrr = sess.run([model.learning_rate_decay_half_op])
                        learning_rate2 = sess.run(model.learning_rate)
                        print("joint_intent_slots_navie==>validation.part.learning_rate1:", learning_rate1," ;learning_rate2:", learning_rate2)
                    else:  # loss is decreasing
                        if eval_loss < best_eval_loss:
                            print("joint_intent_slots_navie==>going to save the model.eval_loss:",eval_loss, ";best_eval_loss:",best_eval_loss)
                            # save model to checkpoint
                            save_path = FLAGS.ckpt_dir + "model.ckpt"
                            saver.save(sess, save_path, global_step=epoch)
                            best_eval_loss = eval_loss
                    previous_eval_loss = eval_loss

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss, test_acc_intent,test_acc_slot = do_eval(sess, model, x_test, y_intent_test, y_slots_test, batch_size)
        print("test_loss:",test_loss,";test_acc_intent:",test_acc_intent,";test_acc_slot:",test_acc_slot)
    pass

# 在验证集上做验证，报告损失、精确度
def do_eval(sess, model, x_valid, y_intent_valid, y_slots_valid, batch_size):
    number_examples = len(x_valid)
    eval_loss, eval_acc_intent,eval_acc_slot, eval_counter = 0.0,0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {model.x: x_valid[start:end], model.dropout_keep_prob: 1.0}
        feed_dict[model.y_intent] = y_intent_valid[start:end]
        feed_dict[model.y_slots] = y_slots_valid[start:end]
        curr_eval_loss, curr_eval_acc_intent,curr_eval_acc_slot = sess.run([model.loss_val, model.accuracy_intent,model.accuracy_slot],feed_dict)
        eval_loss, eval_acc_intent,eval_acc_slot, eval_counter = eval_loss + curr_eval_loss, eval_acc_intent +curr_eval_acc_intent , eval_acc_slot+curr_eval_acc_slot,eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc_intent / float(eval_counter),eval_acc_slot / float(eval_counter)


if __name__ == "__main__":
    tf.app.run()