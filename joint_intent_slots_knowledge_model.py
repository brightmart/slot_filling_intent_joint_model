# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# seq2seq_attention: 1.word embedding 2.encoder 3.decoder(optional with attention). for more detail, please check:Neural Machine Translation By Jointly Learning to Align And Translate
import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
import random
import copy
import os

class joint_knowledge_model:
    def __init__(self, intent_num_classes, learning_rate, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,hidden_size, sequence_length_batch,slots_num_classes,is_training,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=3.0,l2_lambda=0.0001,use_hidden_states_slots=True,filter_sizes=[1,2,3,4,5],num_filters=128):
        """init all hyperparameter here"""
        # set hyperparamter
        self.intent_num_classes = intent_num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.80) #0.5
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients
        self.l2_lambda=l2_lambda
        self.sequence_length_batch=sequence_length_batch
        self.slots_num_classes=slots_num_classes
        self.use_hidden_states_slots=use_hidden_states_slots
        #below is for TextCNN
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.num_filters_total = self.num_filters * len(filter_sizes)

        self.x = tf.placeholder(tf.int32, [None, self.sequence_length], name="x")
        self.y_slots = tf.placeholder(tf.int32, [None, self.sequence_length],name="y_slots")
        self.y_intent = tf.placeholder(tf.int32, [None],name="y_intent")
        self.input_knowledges = tf.placeholder(tf.int32, [None, self.sequence_length],name="input_knowledges")  #nput_knowledges

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.encoder()
        self.logits_slots = self.inference_slot() #[none,sequence_length,slots_num_classes]

        self.logits_intent = self.inference_intent() #[none,intent_num_classes]

        self.predictions_intent = tf.argmax(self.logits_intent, axis=1,name="predictions_intent")  # [batch_size]
        self.predictions_slots = tf.argmax(self.logits_slots, axis=2, name="predictions_slots") #[batch_size,slots_num_classes]

        correct_prediction_intent = tf.equal(tf.cast(self.predictions_intent, tf.int32),self.y_intent)  # [batch_size]
        self.accuracy_intent = tf.reduce_mean(tf.cast(correct_prediction_intent, tf.float32), name="accuracy_intent")  # shape=()

        correct_prediction_slot = tf.equal(tf.cast(self.predictions_slots, tf.int32),self.y_slots)  #[batch_size, self.sequence_length]
        self.accuracy_slot = tf.reduce_mean(tf.cast(correct_prediction_slot, tf.float32), name="accuracy_slot") # shape=()
        if not is_training:
            return
        self.loss_val = self.loss_seq2seq()
        self.train_op = self.train()

    def encoder(self):
        """ 1.Word embedding. 2.Encoder with GRU """
        # 1.word embedding
        embedded_words = tf.nn.embedding_lookup(self.Embedding,self.x)  # [None, self.sequence_length, self.embed_size]
        # 2.encode with bi-directional GRU
        fw_cell =tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True) #rnn_cell.LSTMCell
        bw_cell =tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        #fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob);bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
        bi_outputs, self.bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_words, dtype=tf.float32,#sequence_length: size `[batch_size]`,containing the actual lengths for each of the sequences in the batch
                                                          sequence_length=self.sequence_length_batch, time_major=False, swap_memory=True)
        self.inputs_representation=tf.concat([bi_outputs[0],bi_outputs[1]],-1) #should be:[none, self.sequence_length,self.hidden_size*2]

        self.input_knowledges_embedding = tf.nn.embedding_lookup(self.Embedding_slot_label,self.y_slots) #[batch_size,sequence_length,hidden_size]


    def inference_intent(self): #intent
        with tf.variable_scope("hidden_layer"):
            #self.hidden_states_slots:[none,sequence_length,hidden_size]
            #if self.use_hidden_states_slots: #use hidden state of slots as additional features
            #    features=tf.concat([self.hidden_states_slots,self.inputs_representation],axis=2) #[none,sequence_length,hidden_size*3]
            #else:
            #features=self.inputs_representation
            #features = tf.concat([self.input_knowledges_embedding, self.inputs_representation],axis=2)  #TODO ADD [none,sequence_length,hidden_size*3]
            hidden_states=self.conv_layer()
            #inputs_representation=tf.reduce_max(self.inputs_representation,axis=1) #features[none,hidden_size*2]
            input_knowledges=tf.reduce_max(self.input_knowledges_embedding,axis=2) #[batch_size,hidden_size]
            input_knowledges=tf.layers.dense(input_knowledges,self.hidden_size,activation=tf.nn.tanh) #[none,hidden_size]
            input_knowledges = tf.nn.dropout(input_knowledges,keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]

            features = tf.concat([hidden_states,input_knowledges ], axis=1)

            #hidden_states=tf.layers.dense(cnn_feature,self.hidden_size,activation=tf.nn.tanh) #[none,hidden_size]
        logits = tf.matmul(features, self.W_projection_intent) + self.b_projection_intent #[none,intent_num_classes]
        return logits



    def inference_slot(self): #slot
        logits = [] #self.inputs_representation：[none, self.sequence_length,self.hidden_size*2]
        hidden_states_list=[]
        for i in range(self.sequence_length):
            feature=self.inputs_representation[:,i,:] #[none,self.hidden_size*2]
            #knowledge=self.input_knowledges_embedding[:,i,:] ##[none,self.hidden_size]
            #features = tf.concat([feature,knowledge], axis=2)
            hidden_states = tf.layers.dense(feature, self.hidden_size, activation=tf.nn.tanh) #[none,hidden_size]
            output=tf.matmul(hidden_states, self.W_projection_slot) + self.b_projection_slot #[none,slots_num_classes]
            logits.append(output)
            hidden_states_list.append(hidden_states)
        #logits is a list. each element is:[none,slots_num_classes]
        logits=tf.stack(logits,axis=1) #[none,sequence_length,slots_num_classes]
        self.hidden_states_slots=tf.stack(hidden_states_list,axis=1) #[none,sequence_length,hidden_size]
        return logits

    def conv_layer(self):
        dimension=self.inputs_representation.get_shape().as_list()[-1]
        self.inputs_representation_expanded = tf.expand_dims(self.inputs_representation,-1)  # [None,sentence_length,hidden_size*2,1]. expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                filter = tf.get_variable("filter-%s" % filter_size,[filter_size, dimension, 1, self.num_filters],initializer=self.initializer)
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width, channels];output:4-D
                conv = tf.nn.conv2d(self.inputs_representation_expanded, filter, strides=[1, 1, 1, 1],padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  #bias.
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID',name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        self.h_pool = tf.concat(pooled_outputs,-1)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filters_total])  # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            cnn_feature = tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]

        return cnn_feature

    def loss_seq2seq(self):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, intent_num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            loss_slot = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_slots, logits=(self.logits_slots+self.epsilon()))  #[none, self.sequence_length]. A `Tensor` of the same shape as `labels`
            loss_slot=tf.reduce_sum(loss_slot,axis=1)/self.sequence_length #loss_batch:[batch_size]
            loss_slot=tf.reduce_mean(loss_slot) #scalar

            loss_intent= tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_intent, logits=(self.logits_intent+self.epsilon())) #[batch_size].#A `Tensor` of the same shape as `labels`
            loss_intent=tf.reduce_mean(loss_intent) #scalar

            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            weights_intent=tf.nn.sigmoid(tf.cast(self.global_step/1000,dtype=tf.float32))/2
            loss = (1.0-weights_intent)*loss_slot+weights_intent*loss_intent + l2_losses
            return loss

    def epsilon(self,dtype=tf.float32):
        if dtype is tf.float16:
            return 1e-7
        else:
            return 1e-10

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding_slot_label = tf.get_variable("Embedding_slot_label", shape=[self.slots_num_classes, self.embed_size],dtype=tf.float32) #,initializer=self.initializer
        with tf.name_scope("projection"):  # embedding matrix
            # w projection slot is used for slot. slots_num_classes means how many slots name totally.
            self.W_projection_slot = tf.get_variable("W_projection_slot", shape=[self.hidden_size, self.slots_num_classes],initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection_slot = tf.get_variable("b_projection_slot", shape=[self.slots_num_classes])
            # w projection is used for intent. intent_num_classes mean target side classes.
            self.W_projection_intent = tf.get_variable("W_projection_intent", shape=[self.num_filters_total+self.hidden_size, self.intent_num_classes],initializer=self.initializer)  #[self.hidden_size,self.vocab_size]
            self.b_projection_intent = tf.get_variable("b_projection_intent", shape=[self.intent_num_classes])


# test started: for slot_filling part,for each element,learn to predict whether its value with previous and next element(let's say,sub_sum) is great than a threshold;
# for intent, predict total elements that its sub_sum greater than threshold.
def train():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    intent_num_classes = 100 #additional two classes:one is for _GO, another is for _END
    learning_rate = 0.0001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 300
    embed_size = 1000 #100
    hidden_size = 1000
    is_training = True
    dropout_keep_prob = 0.5  # 0.5 #num_sentences
    decoder_sent_length=6
    l2_lambda=0.0001
    slots_num_classes=2
    sequence_length_batch=[sequence_length]*batch_size
    model = joint_knowledge_model(intent_num_classes, learning_rate, decay_steps, decay_rate, sequence_length,
                                    vocab_size, embed_size,hidden_size, sequence_length_batch,slots_num_classes,is_training,l2_lambda=l2_lambda)
    ckpt_dir = 'checkpoint_dmn/dummy_test/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1500): #1500
            # prepare data
            label_list=get_unique_labels(sequence_length)
            encoder_input = np.array([label_list],dtype=np.int32) #[2,3,4,5,6]
            input_knowledges=np.array([get_knowledge_list(label_list,slots_num_classes)],dtype=np.int32)
            target_list=get_target_from_list(label_list)
            decoder_slot_input=np.array([[0]+target_list],dtype=np.int32) #[[0,2,3,4,5,6]] #'0' is used to represent start token.
            decoder_slot_output=np.array([target_list+[1]],dtype=np.int32) #[[2,3,4,5,6,1]] #'1' is used to represent end token.
            decoder_intent=[np.sum(target_list)]
            # feed the data and run.
            loss, predictions_intent,predictions_slots,accuracy_intent, accuracy_slot,_ = sess.run([model.loss_val,model.predictions_intent,
                                                                           model.predictions_slots,model.accuracy_intent,model.accuracy_slot, model.train_op],
                                                     feed_dict={model.encoder_input:encoder_input,
                                                                model.decoder_slot_input:decoder_slot_input,
                                                                model.decoder_slot_output: decoder_slot_output,
                                                                model.decoder_intent:decoder_intent,
                                                                model.input_knowledges: input_knowledges,
                                                                model.dropout_keep_prob: dropout_keep_prob})
            if i%1000==0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess,save_path,global_step=i)

            print(i,";loss:", loss, ";inputX:",label_list,";dec_slot_output:", list(decoder_slot_output[0]), ";preds_slots:", list(predictions_slots[0]),
                  ";dec_intent:",list(decoder_intent),";pred_intent:",list(predictions_intent),"acc_intent:",accuracy_intent,"acc_slot:",accuracy_slot)

def predict():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    intent_num_classes =100  # additional two classes:one is for _GO, another is for _END
    learning_rate = 0.0001
    batch_size = 1
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 300
    embed_size = 1000
    hidden_size = 1000
    is_training = False #THIS IS DIFFERENT FROM TRAIN()
    dropout_keep_prob = 1.0
    decoder_sent_length = 6
    l2_lambda = 0.0001
    slots_num_classes=2
    sequence_length_batch=[sequence_length]*batch_size
    model = joint_knowledge_model(intent_num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                                    vocab_size, embed_size, hidden_size, sequence_length_batch,slots_num_classes,is_training,
                                    decoder_sent_length=decoder_sent_length, l2_lambda=l2_lambda)
    ckpt_dir = 'checkpoint_dmn/dummy_test/'
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
        for i in range(100):
            # prepare data
            label_list=get_unique_labels(sequence_length)
            encoder_input = np.array([label_list],dtype=np.int32) #[2,3,4,5,6]
            input_knowledges=np.array([get_knowledge_list(label_list,slots_num_classes)],dtype=np.int32)
            decoder_slot_input=np.array([[0]+[0]*len(label_list)],dtype=np.int32) #[[0,2,3,4,5,6]] #'0' is used to represent start token.
            # feed the data and run.
            predictions_intent,predictions_slots = sess.run([model.predictions_intent,model.predictions_slots],
                                                     feed_dict={model.encoder_input:encoder_input,
                                                                model.decoder_slot_input:decoder_slot_input,
                                                                model.input_knowledges: input_knowledges,
                                                                model.dropout_keep_prob: dropout_keep_prob})
            target_list=get_target_from_list(label_list)
            decoder_slot_output=np.array([target_list+[1]],dtype=np.int32) #[[2,3,4,5,6,1]] #'1' is used to represent end token.
            decoder_intent=[np.sum(target_list)]
            print(i, "input x:", label_list,";decoder_slot_output[right]:", decoder_slot_output,
                  "pred_slots:",predictions_slots,"decoder_intent:",decoder_intent,"pred_intent:", predictions_intent)


def get_unique_labels(size):
    x=[2,3,4,5,6,7,8,9]
    random.shuffle(x)
    x=x[0:size]
    return x

def get_target_from_list(list, threshold=15):
    #result_list = [int(e % 2 == 0) for e in list]
    result_list=[]
    previous=0
    next=0
    length=len(list)
    for i,element in enumerate(list):
        if i-1<0:
            previous=0
        else:
            previous=list[i-1]
        if i+1>length-1:
            next=0
        else:
            next=list[i+1]
        sub_sum=np.sum(previous+element+next)
        value=1 if  sub_sum>=threshold else 0
        result_list.append(value)
    return result_list

#result=get_target_from_list([2,5,7,4,9])
#print("result:",result)

def get_knowledge_list(label_list,slots_voc_size,threshold=15):
    result_list=[] #[0]*slots_voc_size
    singel_threshold=threshold/3
    for i,element in enumerate(label_list):
        sub_list = 1 if element>=singel_threshold else 0
        result_list.append(sub_list)
    result_list.append(0)
    return result_list

def get_knowledge_listOLD(label_list,slots_voc_size,threshold=15):
    result_list=[] #[0]*slots_voc_size
    singel_threshold=5
    for i,element in enumerate(label_list):
        sub_list = [1 if element>=singel_threshold else 0 for e in range(slots_voc_size)]
        result_list.append(sub_list)
    result_list.append([ 0 for e in  range(slots_voc_size)])
    return result_list

#result=get_knowledge_list([4,3,5,2,6],2)
#print("result:",result)

#train()
#predict()
