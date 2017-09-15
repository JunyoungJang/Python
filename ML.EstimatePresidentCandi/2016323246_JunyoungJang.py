# -*- coding: utf-8 -*-
from konlpy.tag import *
import numpy as np
import os

def voca_list(filename, listname):
    while 1:
        voca = filename.readline()
        line_parse = kkma.pos(voca)
        for i in line_parse:
            if i[1] == u'SW':
                if i[0] in [u'NaN', u'NaN']:
                    listname.append(i[0])
            if i[1] in list_tag:
                listname.append(i[0])
        if not voca:
            break
    return listname

def naive_prob(test, train, numList):
    counter = 0
    list_count = []
    for i in test:
        for j in range(len(train)):
            if i == train[j]:
                counter = counter + 1
        list_count.append(counter)
        counter = 0
    list_naive = []
    for i in range(len(list_count)):
        list_naive.append((list_count[i] + 1) / float(len(train) + numList))
    result = 1
    for i in range(len(list_naive)):
        result *= float(round(list_naive[i], 6))
    return float(result) * float(1.0 / 3.0)

def cal_Posprob(name):
    mer_pos_name = 'data2016323246/%s_pos_merge.txt' % name
    mer_neg_name = 'data2016323246/%s_neg_merge.txt' % name
    s_test = 'data2016323246/%s.txt' % name
    if not os.path.exists(mer_pos_name):
        s_pos_base = 'data_jjy/positive.txt'
        s_neg_base = 'data_jjy/negative.txt'
        s_pos = 'data_jjy/%s_pos.txt' % name
        s_neg = 'data_jjy/%s_neg.txt' % name

        voca_pos_base = []
        voca_neg_base = []
        f_pos_base = open(s_pos_base, 'r', encoding='UTF8')
        f_neg_base = open(s_neg_base, 'r', encoding='UTF8')
        voca_pos_base = voca_list(f_pos_base, voca_pos_base)
        voca_neg_base = voca_list(f_neg_base, voca_neg_base)

        voca_pos_candi = []
        voca_neg_candi = []
        f_pos = open(s_pos, 'r', encoding='UTF8')
        f_neg = open(s_neg, 'r', encoding='UTF8')
        voca_pos_candi = voca_list(f_pos, voca_pos_candi)
        voca_neg_candi = voca_list(f_neg, voca_neg_candi)

        voca_positive = list(set(voca_pos_candi + voca_pos_base))
        voca_negative = list(set(voca_neg_candi + voca_neg_base))

        fm_pos = open(mer_pos_name, 'w', encoding='UTF8')
        fm_neg = open(mer_neg_name, 'w', encoding='UTF8')
        for i in range(len(set(voca_positive))):
            voca_positive_str = '%s\n' % voca_positive[i]
            voca_negative_str = '%s\n' % voca_negative[i]
            fm_pos.write(voca_positive_str)
            fm_neg.write(voca_negative_str)
        fm_pos.close()
        fm_neg.close()
    else:
        f_pos = open(mer_pos_name, 'r', encoding='UTF8')
        f_neg = open(mer_neg_name, 'r', encoding='UTF8')
        voca_positive = []
        voca_negative = []
        voca_positive = voca_list(f_pos, voca_positive)
        voca_negative = voca_list(f_neg, voca_negative)
    numList = len(set(voca_positive)) + len(set(voca_negative))
    f_test = open(s_test, 'r', encoding='UTF8')
    non_blank_count = 0
    for line in f_test:
        if line.strip():
            non_blank_count = non_blank_count + 1

    Pos = 0
    f_test = open(s_test, 'r', encoding='UTF8')
    for ind in range(non_blank_count):
        test_s = f_test.readline()
        test_list = kkma.pos(test_s)
        test_output = []
        for i in test_list:
            if i[1] == u'SW':
                if i[0] in [u'NaN', u'NaN']:
                    test_output.append(i[0])
            if i[1] in list_tag:
                test_output.append(i[0])
        result_pos = naive_prob(test_output, voca_positive, numList)
        result_neg = naive_prob(test_output, voca_negative, numList)
        if (result_pos > result_neg):
            Pos = Pos + 1
        notice = '[%d/%d] = %d , %f (%s)' % (ind+1, non_blank_count, Pos, Pos/non_blank_count, name)
        print(notice)
    Prop = Pos/non_blank_count
    return Prop, Pos, non_blank_count

kkma = Kkma()
list_tag = [u'NNG', u'VV', u'VA', u'VXV', u'UN']

MoonProp, MoonPos, MoonRange = cal_Posprob('moon')
RedProp, RedPos, RedRange = cal_Posprob('red')
V3Prop, V3Pos, V3Range = cal_Posprob('v3')

TProp = MoonProp + RedProp + V3Prop
SetPropTwitter = (np.array([MoonProp, RedProp, V3Prop])/TProp)*100
SetPropGoogle = np.array([45, 35, 20])
SetPropGallup = np.array([51, 24, 25])

Result = SetPropTwitter*0.3 + SetPropGoogle*0.4 + SetPropGallup*0.3
ResultPrint0 = 'Moon = %.2f,   Red = %.2f,   V3 = %.2f'%(SetPropTwitter[0], SetPropTwitter[1], SetPropTwitter[2])
ResultPrint1 = 'Moon = %.2f,   Red = %.2f,   V3 = %.2f'%(SetPropGoogle[0], SetPropGoogle[1], SetPropGoogle[2])
ResultPrint2 = 'Moon = %.2f,   Red = %.2f,   V3 = %.2f'%(SetPropGallup[0], SetPropGallup[1], SetPropGallup[2])
ResultPrint3 = 'Moon = %.2f,   Red = %.2f,   V3 = %.2f'%(Result[0], Result[1], Result[2])
print('----------------------------------------------------')
print('Twitter (30%):')
print(ResultPrint0)
print('Google (40%):')
print(ResultPrint1)
print('Gallup (30%):')
print(ResultPrint2)
print('Therefore, ')
print(ResultPrint3)
print('----------------------------------------------------')