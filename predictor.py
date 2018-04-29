# coding=utf-8
import os
import sys
import time
import math
import random
from copy import deepcopy
import pack

'''
输入：
initDate:我们设定的初始日期，即我们训练数据中开始的那天
currentDate:需要进行编码的日期
输出：编码结束之后的日期

进行编码日期，方便后期的处理
'''


def EncodingDate(initDate, encodingDate):
    initDate = initDate.split('-')
    encodingDate = encodingDate.split('-')
    init_t = (int(initDate[0]), int(initDate[1]), int(initDate[2]), 0, 0, 0, 0, 0, 0)
    current_t = (int(encodingDate[0]), int(encodingDate[1]), int(encodingDate[2]), 0, 0, 0, 0, 0, 0)
    # print(init_t,current_t)
    encodingDay = int((time.mktime(current_t) - time.mktime(init_t)) / 3600 / 24)
    return encodingDay


'''
输入：cfgTxt:配置文件读取后的list形式数组
     initDate:训练数据集中的第一天
输出：serverCfg:服务器配置，格式如['56 128 1200']
     flavorCfg:提供的虚拟机种类和规格，格式如['15','flavor1 1 1024','flavor2 1 2048',...]
     boxTarget:装箱需要优化的对象，初赛有CPU和MEN两种，单目标。格式如['CPU']
     predictDate:需要进行预测的日期时间段，格式如['2015-02-20','2015-02-27']
预处理配置文件，输入为读取文件后的list形式文件
'''


def PreProcessCfg(cfgTxt,initDate):
    serverCfg = []
    flavorCfg = []
    predictDate = []
    serverNames=[]
    current = 0
    # print([cfgTxt])
    server_number=int(cfgTxt[0].split('\r\n')[0])

    for i in range(1,server_number+1):
        cur_server_cfg = cfgTxt[i].split('\r\n')[0]
        cur_server_cfg = cur_server_cfg.split(' ')
        serverNames.append(cur_server_cfg[0])
        server_i = []
        for i in range(1, len(cur_server_cfg) - 1):
            # print(serverCfg)
            server_i.append(int(float(cur_server_cfg[i])))
        serverCfg.append(server_i)
    move_pace = server_number+1
    while cfgTxt[move_pace]=='\r\n':
        move_pace+=1
    # print (cfgTxt[move_pace])
    flavor_num=int(cfgTxt[move_pace].split('\r\n')[0])
    for i in range(move_pace,move_pace+flavor_num+1):
        flavorCfg.append(cfgTxt[i].split('\r\n')[0])
    move_pace+=flavor_num+1
    while cfgTxt[move_pace]=='\r\n':
        move_pace+=1
    for i in range(move_pace,move_pace+2):
        currentDate = cfgTxt[i].split('\r\n')[0]
        cur_day = EncodingDate(initDate, currentDate.split(' ')[0])
        if (currentDate.split(' ')[1] == '23:59:59'):
            cur_day += 1
        predictDate.append(cur_day)
    flavor_name = []
    flavor_type = {}
    print ('flavorCfg:',flavorCfg)
    for i in range(1, len(flavorCfg)):
        current_flavor_name = flavorCfg[i].split(' ')[0]
        flavor_name.append(current_flavor_name)
        # print ('flavorCfg[i]:',flavorCfg[i])
        flavor_type[current_flavor_name] = [int(flavorCfg[i].split(' ')[1]), int(flavorCfg[i].split(' ')[2])]

    return serverCfg, flavorCfg, serverNames, predictDate, flavor_name, flavor_type

'''
输入：配置文件读取后的list形式一维数组
输出：initialDate:输出用于日期编码的第一天,数据格式如:'2015-01-01'
     processedData:二维数组[flavor,day]，预处理后得到的数据，数据格式如：[['flavor9', 15], ['flavor9', 15], ...]
'''


def PreProcessTrainningData(dataTable):
    processedData = []
    initialDate = dataTable[0].split('\t')[2].split(' ')[0]

    for i in range(len(dataTable)):
        currentDate = dataTable[i].split('\t')[2].split(' ')[0]
        row_i = [dataTable[i].split('\t')[1], EncodingDate(initialDate, currentDate)]
        processedData.append(row_i)
    return initialDate, processedData


'''
输入:preProcessedData:二维数组[flavor,day]，预处理后得到的数据，数据格式如：[['flavor9', 15], ['flavor9', 15], ...]
    flavorCfg:提供的虚拟机种类和规格，格式如['15','flavor1 1 1024','flavor2 1 2048',...]
输出:trainningMatrix:获取二维矩阵，横轴为一个flavor对应一系列天数的个数，纵轴为一天对应的各个flavor使用个数

获取训练数据的二维表。
'''


def generateMatrix(preprocessedData, flavorCfg):
    maxDay = preprocessedData[-1][1]
    numOfFlavors = int(flavorCfg[0])

    trainningMatrix = [[0] * (maxDay + 1) for i in range(numOfFlavors)]
    flavorMap = dict()
    for i in range(numOfFlavors):
        flavorMap[flavorCfg[i + 1].split(' ')[0]] = i

    for data_i in preprocessedData:
        if data_i[0] in flavorMap.keys():
            row_i = flavorMap[data_i[0]]
            trainningMatrix[row_i][data_i[1]] = trainningMatrix[row_i][data_i[1]] + 1
        else:
            continue
    return trainningMatrix


'''
输入：
    trainingDataPath:训练数据的路径
    cfgPath:配置数据的路径
输出：
    trainnningTable:获取二维表，横轴为一个flavor对应一系列天数的个数，纵轴为一天对应的各个flavor使用个数
    initDate:训练数据开始的日期，如 [2015-1-1]
    serverCfg:服务器的配置[CPU,MEM,DISK]
    flavorCfg:提供的虚拟机种类和规格，格式如['15','flavor1 1 1024','flavor2 1 2048',...]
    boxTarget:装箱需要优化的对象MEM或CPU,格式如['CPU']
    predictDate:需要预测的日期，格式如[25,32]
'''


def generatePreInfo(trainingTxt, cfgTxt):
    initDate, processedData = PreProcessTrainningData(trainingTxt)
    serverCfg, flavorCfg, serverNames, predictDate, flavorName, flavorType = PreProcessCfg(cfgTxt, initDate)
    trainnningTable = generateMatrix(processedData, flavorCfg)
    return trainnningTable, initDate, serverCfg, flavorType, serverNames, predictDate, flavorName


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print ('file not exist: ' + file_path)
        return None

def sum_of_array(array,axis=0):
    sum_array=[]
    if axis==0:
        for line in array:
            sum_of_line=sum(line)
            sum_array.append(sum_of_line)
    #print(sum_array)
    return sum_array
def moving_average(dataset,slide_window=7):
    if len(dataset[0])<slide_window:
        print("error input function moving_average")
        return
    moving_avg=[]
    for line in dataset:
        cur_moving_avg_line=[]
        sum_of_slide=0
        for i in range(0,len(line)):
            if i<slide_window:
                sum_of_slide+=line[i]
                if i==slide_window-1:
                    cur_moving_avg_line=[sum_of_slide/slide_window]*slide_window
            else:
                sum_of_slide=sum_of_slide-line[i-slide_window]+line[i]
                cur_moving_avg_line.append(sum_of_slide/slide_window)
        moving_avg.append(cur_moving_avg_line)
    return moving_avg

def moving_std(dataset,slide_window=7):
    if len(dataset[0])<slide_window:
        print("error input function moving_std")
        return
    moving_stdev=[]
    moving_avg=moving_average(dataset,slide_window)
    for i in range(len(dataset)):
        sum_of_slide=0
        cur_moving_average=moving_avg[i]
        cur_moving_std_line=[]
        for j in range(0,len(dataset[i])):
            if j<slide_window:
                sum_of_slide+=pow((dataset[i][j]-cur_moving_average[j]),2)
                if j==slide_window-1:
                    cur_moving_std_line=[math.sqrt(sum_of_slide/slide_window)]*slide_window
            else:
                sum_of_slide=sum_of_slide-pow((dataset[i][j-slide_window]-cur_moving_average[j-slide_window]),2)+pow((dataset[i][j]-cur_moving_average[j]),2)
                cur_moving_std_line.append(math.sqrt(sum_of_slide/slide_window))
        moving_stdev.append(cur_moving_std_line)
    return moving_stdev

def acc_sum(dataset):
    acc_res=[]
    for line in dataset:
        cur_acc=[]
        cur_sum=0
        for data in line:
            cur_sum+=data
            cur_acc.append(cur_sum)
        acc_res.append(cur_acc)
    return acc_res

def remove_noise(dataset):
    data_moving_avg=moving_average(dataset)
    data_moving_std=moving_std(dataset)
    after_remove_noise=[]
    for i in range(len(dataset)):
        cur_row=[]
        for j in range(len(dataset[i])):
            if dataset[i][j]>data_moving_avg[i][j]+2.1*data_moving_std[i][j]:
                if abs(dataset[i][j]-data_moving_avg[i][j])<=3:
                    cur_row.append(dataset[i][j])
                else:
                    addition_weight=max(dataset[i][j]-(data_moving_avg[i][j]+2.1*data_moving_std[i][j]),0.6*data_moving_std[i][j])/2
                    cur_row.append(data_moving_avg[i][j]+0.6*data_moving_std[i][j]+addition_weight)
            else:
                cur_row.append(dataset[i][j])
        after_remove_noise.append(cur_row)
    return after_remove_noise


def dot_product(series_1, series_2):
    if len(series_1) != len(series_2):
        print("dot_product error")
        return
    n = len(series_1)
    product = 0
    for i in range(n):
        product += series_1[i] * series_2[i]
    return product


def post_process_winter(prediction):
    final_prediction = []
    for i in range(len(prediction)):
        if prediction[i] < 0:
            final_prediction.append(0)
        else:
            final_prediction.append(int(round(prediction[i])))
    return final_prediction
    # print(post_process_winter)


def group_by_n(dataset, n):
    dataset_by_n = []
    for i in range(len(dataset)):
        flavor_data = []
        start_pos = len(dataset[i]) % n
        for k in range(len(dataset[i]) // n):
            sum_of_n = 0
            for j in range(start_pos, start_pos + n):
                sum_of_n += dataset[i][j]
            start_pos += n
            flavor_data.append(sum_of_n)
        dataset_by_n.append(flavor_data)
    return dataset_by_n


def convert_list_2_dict(prediction_list,flavorNames):
    prediction_dict = {}
    addition = 0
    res_str = ''
    for i in range(len(flavorNames)):
        addition += prediction_list[i]
        name = flavorNames[i]
        res_str += name + ' ' + str(prediction_list[i]) + '\n'
        prediction_dict[name] = prediction_list[i]
    res_str = str(addition) + '\n' + res_str
    print(res_str)
    return prediction_dict,res_str



def avg_dafa(trainningTable,predictDate,flavorNames):
    def calc_avg(train_tb):
        avg_all=[]
        for i in range(len(train_tb)):
            avg_all.append(sum(train_tb[i])/len(train_tb[i]))
        return  avg_all
    def pro_train_tb(train_tb):
        after_pro_train_tb=[]
        left_bound=0.8
        right_bound=1.2
        for i in range(len(train_tb)):
            flavor_train=[]
            for j in range(len(train_tb[i])):
                cur_res=(left_bound+(right_bound-left_bound)/len(train_tb[i])*j)*train_tb[i][j]
                flavor_train.append(cur_res)
            after_pro_train_tb.append(flavor_train)
        return after_pro_train_tb

    predict_day_span = predictDate[1] - predictDate[0]
    missing_day_span=predictDate[0]-len(trainningTable[0])
    trainning_table_clean = remove_noise(trainningTable)
    pro_train_tb_clean=pro_train_tb(trainning_table_clean)

    avg_train_tb=calc_avg(pro_train_tb_clean)
    para_miss=missing_day_span/8.0
    para_pred=predict_day_span/14.0
    para=1.5+para_miss+para_pred
    prediction=[i*predict_day_span*para for i in avg_train_tb]
    prediction_list= post_process_winter(prediction)
    # prediction_list=prediction

    prediction_dict = {}
    addition = 0
    res_str = ''
    for i in range(len(flavorNames)):
        addition += prediction_list[i]
        name = flavorNames[i]
        res_str += name + ' ' + str(prediction_list[i]) + '\n'
        prediction_dict[name] = prediction_list[i]
    res_str = str(addition) + '\n' + res_str
    print(res_str)
    return prediction_dict, res_str

def smooth(train_tb,alpha):
    train_tb_smooth=[]
    for i in range(len(train_tb)):
        S=[]
        S0=(train_tb[i][0]+train_tb[i][1])/2
        S.append(S0)
        for j in range(1,len(train_tb[i])):
            S_j=alpha*train_tb[i][j]+S[len(S)-1]*(1-alpha)
            S.append(S_j)
        train_tb_smooth.append(S)
    return train_tb_smooth

def stocGradAscent_2D(dataset, numIter=1000, slide_window=7):
    m = len(dataset)
    n = len(dataset[0])
    weights = []
    for k in range(m):
        print ('dataset[k]',dataset[k])
        data_series=[math.log(i) for i in dataset[k]]
        weight_k = [1, 0.1]
        for j in range(numIter):
            alpha = 0.01 / (1.0 + j) + 0.002
            error=0
            error_b=0
            weight_series=[0.85**(len(data_series)-i-1) for i in range(len(data_series))]
            for randIndex in range(n):
                h = weight_k[0] * randIndex + weight_k[1]
                left_bound=0.4
                right_bound=1.6
                error+= (data_series[randIndex] - h)* (randIndex+0.5)*1.0/n*weight_series[randIndex]
                error_b+=(data_series[randIndex] - h)/n*weight_series[randIndex]
                #以下是线性加权
                #error+= (data_series[randIndex] - h)* randIndex*1.0/n*(left_bound+(right_bound-left_bound)*randIndex*1.0/(n-1))
                #error_b+=(data_series[randIndex] - h)/n*(left_bound+(right_bound-left_bound)*randIndex*1.0/(n-1))
            # print(error)
            weight_k[0] = weight_k[0] + alpha * error
            weight_k[1] = weight_k[1] + alpha * error_b
        weight_k[1] = math.exp(weight_k[1])
        weights.append(weight_k)
    return weights

def expon_dafa_lin(trainningTable,predictDate,flavorNames):
    def add_one_to_all(train_tb):
        train_tb_add=deepcopy(train_tb)
        for i in range(len(train_tb_add)):
            for j in range(len(train_tb_add[i])):
                train_tb_add[i][j]+=1
        return train_tb_add

    trainning_table_clean=remove_noise(trainningTable)
    #trainning_table_clean=trainningTable
    predict_day_span = predictDate[1] - predictDate[0]
    missing_day_span = predictDate[0] - len(trainningTable[0])
    n=7
    grouped_train_tb=group_by_n(trainning_table_clean,n)
    #grouped_train_tb=smooth(grouped_train_tb,0.8)
    print ('grouped_train_tb:',grouped_train_tb)

    print('miss_day_span:',missing_day_span,'predict_day_span:',predict_day_span,'len(grouped_train_tb):',len(grouped_train_tb))
    missing_time_stamp=missing_day_span*1.0/n+len(grouped_train_tb[0])-1
    #missing_time_stamp=0*1.0/n+len(grouped_train_tb[0])-1
    predict_time_stamp=(missing_day_span+predict_day_span)*1.0/n+len(grouped_train_tb[0])-1
    print('missing_time_stamp:',missing_time_stamp,'predict_time_stamp:',predict_time_stamp)
    prediction=[]
    numIter=100000

    print ("grouped_train_tb:",grouped_train_tb)
    grouped_train_tb=add_one_to_all(grouped_train_tb)
    print ("grouped_train_tb:", grouped_train_tb)
    weights = stocGradAscent_2D(grouped_train_tb, numIter)

    print ('weights:',weights)
    for i in range(len(trainning_table_clean)):
        # prediction_i=math.exp(weights[i][0])
        b=weights[i][0]*1.03
        a=weights[i][1]
        prediction_i=0
        print ('i:',i,'b:',b,'a:',a)
        if b<0:
            print ('break:','i',i)
            #sum_of_flavor_i=float(sum(trainningTable[i]))
            weight_for_missdayspan=missing_day_span/20.0
            #if missing_day_span<=7:
            #    prediction_i=(float(sum(trainning_table_clean[i]))/len(trainning_table_clean[i])*predict_day_span)*(2+weight_for_missdayspan)
            #else:
            #    prediction_i=(float(sum(trainning_table_clean[i]))/len(trainning_table_clean[i])*predict_day_span)*2.5
            #prediction_i=(sum_of_flavor_i/len(trainningTable[i])*predict_day_span)*(1.6+weight_for_missdayspan)
            
            prediction_i=(float(sum(trainning_table_clean[i]))/len(trainning_table_clean[i])*predict_day_span)*(2.22+weight_for_missdayspan)
            #weight_series=[0.987**(len(trainning_table_clean[i])-k-1) for k in range(len(trainning_table_clean[i]))]
            #prediction_i=dot_product(trainning_table_clean[i],weight_series)/sum(weight_series)*predict_day_span*(2.22+weight_for_missdayspan)
        else:
            print('b*predict_time_stamp',b*predict_time_stamp,'b*missing_time_stamp:',b*missing_time_stamp)
            prediction_missday=(a/b*(math.exp(b*missing_time_stamp)-math.exp(b*(len(grouped_train_tb[0])-1))))
            append_i=0
            if b>0.5:
                append_i=2
            prediction_i=(a/b*(math.exp(b*predict_time_stamp)-math.exp(b*missing_time_stamp)))*1.0+append_i+prediction_missday*1
            if prediction_i>120:
                prediction_i+=2
        if predict_day_span>=10:
            prediction_i+=10
        prediction.append(1*(prediction_i))
    prediction_list = post_process_winter(prediction)
    return convert_list_2_dict(prediction_list,flavorNames)

def test_in_flavors(test_data):
    database=['flavor1','flavor2','flavor3','flavor4','flavor5','flavor6','flavor7','flavor8','flavor9','flavor10','flavor11','flavor12','flavor13','flavor14','flavor15','flavor16','flavor17','flavor18']
    if database.count(test_data)!=1:
        return True
    return False

def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print ('ecs information is none')
        return result
    if input_lines is None:
        print ('input file information is none')
        return result
    trainning_tb,initDate,server_cfg,flavor_type,serverNames,predict_date,flavor_names=generatePreInfo(ecs_lines,input_lines)
    print("predict_date:",predict_date)
    flavor_prediction,res_prediction=expon_dafa_lin(trainning_tb,predict_date,flavor_names)

    res_deploy=pack.pack_flavor(flavor_names,flavor_prediction,flavor_type,server_cfg,serverNames)

    res=res_deploy
    result=[res]

    return result
