# -*- coding:utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import os
import random
import numpy as np

import multiprocessing

"""
import gensim
from gensim.corpora import TextCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
"""

import collections
from collections import Counter
import math
import json

origin_data_file = 'knowledge/wzz_train_data_20171207.txt'#'2c9081a45fe2cab5015fe2d588ae02cb.txt'
output_data_file = 'fitme_data_jhj'
stat_intent_file = 'stat_int_fitme_data_jhj'
stat_slot_file = 'stat_slot_fitme_data_jhj'

slot_vocab_file = 'slot_vocab'


def process_fitme_data():
    output_fp = open(output_data_file, 'w')
    stat_intent_fp = open(stat_intent_file, 'w')
    stat_slot_fp = open(stat_slot_file, 'w')

    a_intents = {}
    s_intents = {}
    u_intents = {}

    a_slots_k = {}
    a_slots_v = {}
    s_slots_k = {}
    s_slots_v = {}
    u_slots = {}

    with open(origin_data_file, 'r') as fp:
        for i, line in enumerate(fp.readlines()):
            line = line.strip()
            dict_inf = json.loads(line)
            actions_inf = dict_inf['actions']

            # event2a need to add 'release_turn'
            # is_event_type = False
            for ii, actor_inf in enumerate(actions_inf):
                # print 'actor: {}\n'.format(actor_inf['actor'])
                if actor_inf['actor'] == 'u':
                    output_slots_inf = actor_inf['slots']
                    for slots_inf in actor_inf['slots']:
                        for slot_name, v in slots_inf.iteritems():
                            if u_slots.has_key(slot_name):
                                u_slots[slot_name] += 1
                            else:
                                u_slots[slot_name] = 1

                    if u_intents.has_key(actor_inf['intent']):
                        u_intents[actor_inf['intent']] += 1
                    else:
                        u_intents[actor_inf['intent']] = 1
                    output_fp.write('[actor:{},target:{}] # {} # {} # slots_inf:{}\n'.format(actor_inf['actor'],
                                                                                             actor_inf['target'],
                                                                                             actor_inf['intent'],
                                                                                             actor_inf['speech'],
                                                                                             output_slots_inf))

                elif actor_inf['actor'] == 's':
                    if actor_inf['target'] == 'a':
                        if not actor_inf['intent'].endswith('的应答'):
                            print
                            'ssssssssss', actor_inf['intent']
                            # is_event_type = True
                            if actions_inf[ii - 1]['actor'] == 'a':
                                output_fp.write('release_turn\n')

                    output_slots_inf = actor_inf['slots']
                    for slots_inf in actor_inf['slots']:
                        for slot_name, v in slots_inf.iteritems():
                            if s_slots_v.has_key(v):
                                s_slots_v[v] += 1
                            else:
                                s_slots_v[v] = 1

                            if s_slots_k.has_key(slot_name):
                                s_slots_k[slot_name] += 1
                            else:
                                s_slots_k[slot_name] = 1

                    if s_intents.has_key(actor_inf['intent']):
                        s_intents[actor_inf['intent']] += 1
                    else:
                        s_intents[actor_inf['intent']] = 1
                    output_fp.write('[actor:{},target:{}] # {} # <NULL> # slots_inf:{}\n'.format(actor_inf['actor'],
                                                                                                 actor_inf['target'],
                                                                                                 actor_inf['intent'],
                                                                                                 output_slots_inf))

                else:  # actor_inf['actor'] = 'a'
                    output_slots_inf = actor_inf['slots']
                    for slots_inf in actor_inf['slots']:
                        for slot_name, v in slots_inf.iteritems():
                            if a_slots_v.has_key(v):
                                a_slots_v[v] += 1
                            else:
                                a_slots_v[v] = 1

                            if a_slots_k.has_key(slot_name):
                                a_slots_k[slot_name] += 1
                            else:
                                a_slots_k[slot_name] = 1

                    if a_intents.has_key(actor_inf['intent']):
                        a_intents[actor_inf['intent']] += 1
                    else:
                        a_intents[actor_inf['intent']] = 1

                    if actor_inf['target'] == 's':
                        output_fp.write('[actor:{},target:{}] # {} # <NULL> # slots_inf:{}\n'.format(actor_inf['actor'],
                                                                                                     actor_inf[
                                                                                                         'target'],
                                                                                                     actor_inf[
                                                                                                         'intent'],
                                                                                                     output_slots_inf))
                    else:
                        output_fp.write('[actor:{},target:{}] # {} # {} # slots_inf:{}\n'.format(actor_inf['actor'],
                                                                                                 actor_inf['target'],
                                                                                                 actor_inf['intent'],
                                                                                                 actor_inf['speech'],
                                                                                                 output_slots_inf))

                # add release_turn
                if ii < (len(actions_inf) - 1):
                    if actor_inf['actor'] == 'a':
                        if actions_inf[ii + 1]['target'] == 'a' and actions_inf[ii + 1]['actor'] != 's':
                            output_fp.write('release_turn\n')

            # add release_turn
            if actions_inf[ii]['actor'] == 'a':
                output_fp.write('release_turn\n')

            output_fp.write('-' * 80 + '\n')

        a_intents = sorted(a_intents.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_intent_fp.write('actor:a, num_intents:{}\n'.format(len(a_intents)))
        for (a_intent, num) in a_intents:
            stat_intent_fp.write('{}    ##    {}\n'.format(a_intent, num))

        s_intents = sorted(s_intents.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_intent_fp.write('actor:s, num_intents:{}\n'.format(len(s_intents)))
        for (s_intent, num) in s_intents:
            stat_intent_fp.write('{}    ##    {}\n'.format(s_intent, num))

        u_intents = sorted(u_intents.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_intent_fp.write('actor:u, num_intents:{}\n'.format(len(u_intents)))
        for (u_intent, num) in u_intents:
            stat_intent_fp.write('{}    ##   {}\n'.format(u_intent, num))

        a_slots_k = sorted(a_slots_k.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_slot_fp.write('actor:a, num_slots_k_name:{}\n'.format(len(a_slots_k)))
        for (a_slot_k, num) in a_slots_k:
            stat_slot_fp.write('{}   ##    {}\n'.format(a_slot_k, num))

        a_slots_v = sorted(a_slots_v.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_slot_fp.write('actor:a, num_slots_v_name:{}\n'.format(len(a_slots_v)))
        for (a_slot_v, num) in a_slots_v:
            stat_slot_fp.write('{}   ##    {}\n'.format(a_slot_v, num))

        s_slots_k = sorted(s_slots_k.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_slot_fp.write('actor:s, num_slots_k_name:{}\n'.format(len(s_slots_k)))
        for (s_slot_k, num) in s_slots_k:
            stat_slot_fp.write('{}   ##   {}\n'.format(s_slot_k, num))

        s_slots_v = sorted(s_slots_v.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_slot_fp.write('actor:s, num_slots_v_name:{}\n'.format(len(s_slots_v)))
        for (s_slot_v, num) in s_slots_v:
            stat_slot_fp.write('{}   ##   {}\n'.format(s_slot_v, num))

        u_slots = sorted(u_slots.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        stat_slot_fp.write('actor:u, num_slots_name:{}\n'.format(len(u_slots)))
        for (u_slot, num) in u_slots:
            stat_slot_fp.write('{}   ##   {}\n'.format(u_slot, num))

    output_fp.close()
    stat_intent_fp.close()
    stat_slot_fp.close()


##########################################################################################################################


if __name__ == "__main__":
    process_fitme_data()
