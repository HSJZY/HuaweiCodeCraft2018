# coding=utf-8
import time
import random
import copy
#单个箱子CPU和内存空间剩余率的乘积,并计算所剩空间大小
def score_singlebox(cpu_men_total, box_type):
    result = 1
    surplus = []
    for index in range(len(cpu_men_total)):
        temp_result =  1 - float(cpu_men_total[index]) / box_type[index]
        surplus.append(box_type[index]-cpu_men_total[index])
        result *= temp_result
    return result, surplus
#返回剩余利用率乘积和剩余数数量 每个为二维

#将当前flavor的大小添加到cpu和men
def cpu_men_sum(sum_result, flavor_addition):
    #v0 as inplace
    for index in range(len(sum_result)):
        sum_result[index] = sum_result[index] + flavor_addition[index]

#将传入的所有flavor序列相加 encode_prediction=[1,2,3,...]
def sum_flavor(encode_prediction, flavor_type):
    sum_result = [0] * len(flavor_type[1])
    for flavor_number in encode_prediction:
        cpu_men_sum(sum_result, flavor_type[flavor_number])
    return sum_result
#返回二维的CPU与men总数

#对单个flavor序列进行分配装箱 
def allocate_single_list(single_list, box_type, flavor_types):
      
    def sum_check_result(results):
        return sum(results)#返回True的个数
    
    def check_boxsingle(box_limit, target_limitation):#检查单个箱子的使用是否超过极限
        for index in range(len(box_limit)):
            if box_limit[index] > target_limitation[index]:
#                print("current:", index)
                return True
        return False
    
    def check_boxlimit(box_limit_all, box_type):#检查所有箱子序列中的箱子的使用是否超限
        check_results = []#box_limit_all=[[2, 4096], [2, 4096], [2, 4096]...]为当前flavor序列中放入各类型的箱子
        for index, box_limit in enumerate(box_limit_all):#index 默认箱子规则为 0,1,2
#            print("box_limit_all:", box_limit_all)
#            print("box_type:", box_type[index])
            check_results.append(check_boxsingle(box_limit, box_type[index]))
        return check_results  
#            print("check_results:", check_results)  
    
    def sum_cm(box_limit_single, single_flavor_size):#把当前flavor的大小加到当前箱子里
        #box_limit_all
        for index in range(len(box_limit_single)):
            box_limit_single[index] =box_limit_single[index] + single_flavor_size[index]  
    #checked_results, limitation_temp, flavor_limitation, index_point_temp, index             
    def single_flavor_add(checked_results, box_limit_all, single_flavor_size, across_points, point):
#        print("box_limit_all:", box_limit_all)
#        print("across_points:", end_points)
        for index, result in enumerate(checked_results):#三个箱子中如当前箱子还未填满，就把当前flavor放入到箱子中
            if (result == False):
                sum_cm(box_limit_all[index], single_flavor_size)
                across_points[index] = point#记录当前放到箱子flavor的位置
    
    #评价单个箱子的利用率，为cup利用率加上men利用率           
    def evaluate_single(box_limit_single, box_type):
        result = 1
        for index in range(len(box_limit_single)):
            result += (float(box_limit_single[index])/ box_type[index])*100
#            print("result:", result)
        return result
    #(box_limit_all, box_type, across_points, flavor_types, single_list)
    
    def reduce_last_flavor(all_limit, last_number):#去掉箱子最后一个flavor，因为该flavor使得箱子使用超限了
        for index in range(len(all_limit)):
            all_limit[index] = all_limit[index] - last_number[index]  
      
    def pack_single_box(box_limit_all, box_type, across_points, flavor_types, single_list):#为三种类型的箱子选择效率最高的
#        print("box_limit_all:", box_limit_all)
        results = []
        for index in range(len(box_limit_all)):
            reduce_last_flavor(box_limit_all[index], flavor_types[single_list[across_points[index]]])
            #把总数减去最后一个溢出的
            results.append(evaluate_single(box_limit_all[index], box_type[index]))
#        print("results:", results)
        return results.index(max(results)), across_points[results.index(max(results))]
        #返回效率最高的箱子的效率的索引，及最后一次访问的位置
    def pack_last_box(box_limit_all, box_type, checked_results):#为最后一个箱子选择
        results = []
        for index, check in enumerate(checked_results):
            if check == True:
                results.append(0)#如何满了就不考虑，评分为零
            else:
                results.append(evaluate_single(box_limit_all[index], box_type[index]))
        return results.index(max(results))
    
    box_limit_all = [[0] * len(box_type[0]) for i in range(len(box_type))]#3X2维
    across_points = [0] * len(box_type)#3个元素
    checked_results = [False] * len(box_type)
    pack_result = []#每个箱子flavor的序列
    box_result = []#箱子序列
    start_point = 0
    last_point = len(single_list) 
    index = start_point
    while(True):
    #把single list的值往3个箱子里面装，满了停止    
        while(sum_check_result(checked_results) != len(checked_results)):#检查True的个数

#            print("checked_results:", checked_results)
            single_flavor_size = flavor_types[single_list[index]]
            single_flavor_add(checked_results, box_limit_all, single_flavor_size, across_points, index)

            checked_results = check_boxlimit(box_limit_all, box_type)
#            print("checked_results:", checked_results)
            index += 1

            if(index == last_point):
                break
        if(sum_check_result(checked_results) == len(checked_results)):
#            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            good_box, end_point = pack_single_box(box_limit_all, box_type, across_points, flavor_types, single_list)
#            print("good_box:", good_box)是箱子类型的关键词，减掉最后一个箱子的位置
#            print("end_point:", end_point)是flavor切断点的位置
#            input()
            pack_result.append(single_list[start_point:end_point])#存放所有箱子所装的flavor序列
            box_result.append(good_box)#存放所有箱子的对应类型号序列
            box_limit_all = [[lim for lim in flavor_types[single_list[end_point]]] for i in range(len(box_type))]
            across_points = [end_point] * len(box_type)
            start_point = end_point
            index = end_point + 1
#            print("start_point:", start_point)
#            print("index:", index)
            checked_results = check_boxlimit(box_limit_all, box_type)
        if(index == last_point):
            checked_results = check_boxlimit(box_limit_all, box_type)
            good_box = pack_last_box(box_limit_all, box_type, checked_results)#返回分数最高箱子的类型号码
            pack_result.append(single_list[start_point:])
            box_result.append(good_box)
            break
    return pack_result, box_result #返回装箱情况及对应的箱子类型
#二号装箱算法
def second_pack(prediction, box_type, flavor_type, number_population, start_time, run_time, good_boy):
    def change_type_prediction(prediction):#改变预测的表达类型
        list_flavor = []
        for key, value in prediction.items():
            for i in range(value):
                list_flavor.append(key)
        return list_flavor#返回预测的flavor序列 list_flavor=[1,2,1,2,3,..]
    #由预测的flavor序列产生第一个种群
    def first_population(list_flavor, number_population):
        first_population = []
        for i in range(number_population):
            random.shuffle(list_flavor)
            first_population.append(copy.copy(list_flavor))    
        return first_population
    #单个flavor序列所用箱子的装箱评分
    def single_listbox_score(total_cm, single_list_box, box_type):
        flavor_summation = sum_flavor(single_list_box, box_type)
#        print("flavor_summation:", flavor_summation)
        result = 0
        for index, summation in enumerate(total_cm):
            result += 5000000 * summation / flavor_summation[index] 
#        print("result:", result)
        return result#返回分数cpu和men利用率乘以50000然后相加
    
    
    def allocate_all_list_score(all_list_flavor, box_type, flavor_type, total_cm):

        evaluation = []
        for singe_list in all_list_flavor:
            #评估 allocate_single_list返回装箱情况及对应的箱子类型
            evaluation.append(
                    (single_listbox_score(total_cm, allocate_single_list(singe_list, box_type, flavor_type)[1], box_type)
                    , singe_list))#allocate_single_list 返回单个flavor序列装箱情况和箱子类型序列 箱子类型序列为一维 二维
        return evaluation  #评估 返回分数，和对应的flavor序列
#返回分数cpu和men利用率乘以5000000然后相加

#通过杂交产生对单个flavor序列进行改变，产生下一代
    def produce_childrens(allocate_result, good_boy):
        def exchange_list(single_list_allocate):
            
            father_generation = copy.copy(single_list_allocate)
            generation_size = len(father_generation)
            exchange_point = random.randint(0, generation_size-1) # minus one because endpoint included
            left_len = exchange_point
            right_len = generation_size - 1 - exchange_point

            direction = "left" if left_len < right_len else "right" if left_len > right_len else "center"

            if direction == "left":
                
                exchange_len = random.randint(1, left_len+1)
        #        print("exchange_len:", exchange_len)
                left_end_point = exchange_point + 1
                left_start_point = left_end_point-exchange_len
                
                left_slice = slice(left_start_point, left_end_point)
                
                right_start_point = random.randint(exchange_point+1, generation_size-exchange_len)
                right_end_point = right_start_point+exchange_len
                right_slice = slice(right_start_point, right_end_point)
        #                right_slice = slice()
            elif direction == "right":
                exchange_len = random.randint(1, right_len+1)
        #        print("exchange_len:", exchange_len)        
                right_start_point = exchange_point
                right_end_point = exchange_point + exchange_len
                right_slice = slice(right_start_point, right_end_point)
                
                left_end_point = random.randint(exchange_len, exchange_point)
                left_start_point = left_end_point - exchange_len
                left_slice = slice(left_start_point, left_end_point)
            elif direction == "center":
                return father_generation
            
        #    print("left_slice:", left_slice)
        #    print("right_slice:", right_slice)
            temp = father_generation[left_slice]
            father_generation[left_slice] = father_generation[right_slice]

            father_generation[right_slice] = temp
        #    print("father_generation:", father_generation)
            return father_generation
                
        
        origin_scale = len(allocate_result)
        del(allocate_result[good_boy:])#删除后面的
        produce_group = []
        for i in range(good_boy, origin_scale):
            single_list_allocate = random.sample(allocate_result, 1)[0][1]

            produce_group.append(exchange_list(single_list_allocate))

        return produce_group    
#返回flavor的列表，一个flavor序列为一维     
    def single_server_summation(single_server, flavor_type):
        summation = [0] * len(flavor_type[1])
        for i in single_server:
            summation[0] += flavor_type[i][0]
            summation[1] += flavor_type[i][1]
        return summation                  
#allocate_server返回装箱序列及箱子序号

    list_flavor = change_type_prediction(prediction)#返回encode_prediction =[1,2,4,5,..],flavor具体序列
    total_cm = sum_flavor(list_flavor, flavor_type)#返回所有预测的flavor的CPU与MEN相加
    all_list_flavor = first_population(list_flavor, number_population)
    allocate_result = allocate_all_list_score(all_list_flavor, box_type, flavor_type, total_cm)
    #返评估 返回分数，和对应的flavor序列
    allocate_result.sort(reverse = True)
    while(time.clock()-start_time < run_time):
        childrens = produce_childrens(allocate_result, good_boy)
        allocate_result.extend(allocate_all_list_score(childrens, box_type, flavor_type, total_cm))
        allocate_result.sort(reverse = True)
        print(allocate_result[0][0]/float(100000), len(allocate_single_list(allocate_result[0][1], box_type, flavor_type)[1]))
    #allocate_result 返回得分以及flavor序列
    #return allocate_single_list(allocate_result[0][1], box_type, flavor_type)
    return allocate_result#返回得分以及对应flavor序列
def evaluation_score(flavor_list,box_list, flavor_specification, box_type):
    real_cpu=0
    real_men=0
    total_cpu=0
    total_men=0
    for j in range(0,len(flavor_list)):
        box=flavor_list[j]
        for i in range(0,len(box)):
            real_cpu+=flavor_specification[box[i]][0]
            real_men+=flavor_specification[box[i]][1]
        total_cpu+=box_type[box_list[j]][0]
        total_men+=box_type[box_list[j]][1]
#        print("total_cpu:", total_cpu)
    score=(float(real_cpu)/total_cpu)*0.5*100+(float(real_men)/total_men)*0.5*100
    return score
#选择分数最高的list
def selection_good_list(allocate_result,box_type, flavor_type):
    best_score = allocate_result[0][0]
    list_group = []
    for candidate in allocate_result:
        if candidate[0] == best_score:
            list_group.append([best_score, allocate_single_list(candidate[1], box_type, flavor_type)])
    return list_group #返回分数、装箱flavor序列及箱子规则序列 一个元素三维
#挑选最好的flavor list，cpu和内存剩余量乘积最大的
def evaluate_single(summation, server_limitation):
    result = 1
    surplus = []
    for index in range(len(summation)):
        temp_result =  1 - float(summation[index]) / server_limitation[index]
        surplus.append(server_limitation[index]-summation[index])
#            print("############################################################")
#            print("surplus:", surplus)
#            print("temp_result:", temp_result)
        result *= temp_result
#            print("result:", result)
#            print("############################################################")
    return result, surplus
def summation_flavor(total, flavor):

    for index in range(len(total)):
        total[index] = total[index] + flavor[index]
def summation_flavor_list(prediction, flavor_specification):
    result = [0] * len(flavor_specification[1])
    for pre in prediction:
       summation_flavor(result, flavor_specification[pre])
    return result

def choose_best_list(good_list, flavor_specification, server_limitation):
    index_list = []
    surplueses = []
    idlenesseses = []
    for index_servers, servers in enumerate(good_list):#candidates包含分数、装箱序列与箱子规则序列 三维维
        surpluses = []
        idlenesses = []
        material = zip(servers[1][0], servers[1][1])#material装箱序列和箱子规格为一个元素
        idleness = 0
        for index_server, server in enumerate(material):#每个箱子flavorCPU与men相加
            idleness_temp, surplus = evaluate_single(summation_flavor_list(server[0], flavor_specification), server_limitation[server[1]])
            idleness += idleness_temp#idleness_temp, surplus返回cpu和men利用率相乘和剩余的空间
            idlenesses.append([idleness_temp, index_server])#利用率与箱子号码
            surpluses.append([surplus, index_server])#剩余量与箱子号码
        index_list.append([idleness, servers[1], index_servers])#idleness整个序列的利用率
        surplueses.append(surpluses)#多个符合条件的序列
        idlenesseses.append(idlenesses)
    index_list.sort(reverse=True)
#        print("index_list:", index_list)
    result = index_list[0][1]#返回箱子flavor序列，剩余量（每个箱子剩余量以及箱子对应的位置数），包含利用率（利用率CPU与men乘积，以及对应的箱子位置数）
    return result, surplueses[index_list[0][2]], idlenesseses[index_list[0][2]]
def optimize_box(last_flavor_list, flavor_type, box_type, prediction,times_change):
    maybe_flavor={}
    for key in prediction.keys():
        maybe_flavor[key]=int(round(prediction[key]))
    print("maybe_flavor:", maybe_flavor)
    flavor_list=[]
    flavor_list=last_flavor_list
    #计算单个箱子剩余量
    def claculate_single_surplus(single_box,flavor_type,box_size):
        surplus_vector= []
        for index in range(len(box_size)):
            #一个箱子内实际的虚拟机大小
            single_box_vector=0
            for i in range(len(single_box)):
                single_flavor=single_box[i]
                single_box_vector += flavor_type[single_flavor][index]
#                 print(single_flavor)
            #一个箱子理想的内存大小
            single_box_desire_vector = box_size[index] 
            surplus_vector.append(single_box_desire_vector -single_box_vector)     
        return surplus_vector#返回单个箱子剩余量，二维
#计算所有箱子的剩余量
    def sort_box_vector(flavor_list, flavor_type, box_type):#flavor_list每个箱子flavor序列及对应箱子规则序列
        all_list = []
        surplus_vector=[]
        for index_box, single_box in enumerate(flavor_list[0]):
#             if type(single_box)==type(13):
#                 print ('error',single_box,index_box,flavor_list[1])
#                 input()            
            surplus_vector = claculate_single_surplus(single_box,flavor_type,box_type[flavor_list[1][index_box]])
#            use_ratio 总的内存大小/箱子数X每个箱子数内存 单个箱子的使用率
            all_list.append([surplus_vector,single_box,flavor_list[1][index_box]])#串中每个箱子的剩余数，和箱子对应序列
        return all_list #箱子对应的的剩余数,箱子装箱flavor，箱子的类别
    
    def addition_single_box(single_box_surplus, single_box_flavor, flavor_type, maybe_flavor):#单个箱子的补偿

        def addition_single_flavor(box_surplus, flavor_size):
            matrix = zip(box_surplus, flavor_size)
            result = [box_surplus >= flavor_size for (box_surplus, flavor_size) in matrix]
            judge = sum(result) / len(result)
            if judge == 1:
                return True
            else:
                return False

        maybe_flacors_key=[i for i in maybe_flavor]
        maybe_flacors_key.sort(reverse=True)
        #print('maybe_key:',maybe_flacors_key)
        for flavor in maybe_flacors_key:
#             print(flavor)
            if(addition_single_flavor(single_box_surplus, flavor_type[flavor]) and (maybe_flavor[flavor] > 0)):
                maybe_flavor[flavor] -= 1
                single_box_flavor.append(flavor)
                break
            
    def addition_flavor(all_list, maybe_flavor, box_type, flavor_type):#所有箱子的补偿
        for i in range(0,len(all_list)):
            surplus=all_list[i][0]#某个箱子剩余数及序号
#                print("surplus:", surplus)# surpluses[-1]最后一个元素
            addition_single_box(surplus, all_list[i][1], flavor_type, maybe_flavor)#单个箱子的补偿
#            print("feed_flavor:", feed_flavor_balance)         
        return all_list #包含箱子剩余量，箱子的装箱flavor序列（添加后的）， 对应的箱子类型
    #flavor_list装箱情况及箱子序列的类型
    def calculate_surplus(servers, flavor_type, box_type):#计算箱子的剩余空间
        servers = zip(*servers)
        surpluses = []
        for index, server in enumerate(servers):

            _, surplus = evaluate_single(summation_flavor_list(server[0], flavor_type), box_type[server[1]])    
            surpluses.append([surplus, index])
        return surpluses#剩余量和箱子的索引
    def random_Selection(scores, objects):#剩余量和箱子的索引
#        print("scores:", scores)
#        print("objects:", objects)
        _sum = sum(scores) + 0.00001
        p = [float(_obj) / _sum for _obj in scores]
#        print("p:", p)
        point = random.random()
#        print("point:", point)
        if_ok=True
        for index, _p in enumerate(p):
#            print("sum(p[:index]):", sum(p[:index]))
            if point < sum(p[:index+1]):
                if_ok=False
                return objects[index]
        if if_ok==True: 
            return  0   
    def selection_best_change_box(surplurse):#surplurse 包含箱子剩余数和索引号

            first_box = []
            second_box = []

            for surplus in surplurse:
                first_box.append([surplus[0][0], surplus[-1]])#剩余量和箱子的索引
                second_box.append([surplus[0][1], surplus[-1]])
            first_box.sort()
            second_box.sort()
            first_box = zip(*first_box)#将二者分开
            second_box = zip(*second_box)
            first_object = random_Selection(first_box[0], first_box[1])#返回经过轮盘选择以后的剩余量和箱子索引
            second_object = random_Selection(second_box[0], second_box[1])#返回经过轮盘选择以后的剩余量和箱子索引
 
            return surplurse[first_object], surplurse[second_object]
    def summu_cpu_men(total, flavor):
        result = []
        for index in range(len(total)):
            result.append(total[index]+flavor[index])
        return result

    def calculate_bigger(flavor1, flavor2):
        result = 1
        for index in range(len(flavor1)):
            result *= (flavor1[index] >= flavor2[index])
        return result #first_select_box箱子余量在索引
    def exchange_flavor_twobox(first_select_box, first_box_to, second_select_box, second_box_to, flavor_type):
        def good_flavor(largest, dimension):#输入为交换的余量和对应的箱子序列
            if dimension == 0:
                if largest >= 32:
                    return [16, 17, 18, 13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 16:
                    return [13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 8:
                    return [10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 4:
                    return [7, 8, 9, 4, 5, 6, 1, 2, 3]
                if largest >= 2:
                    return [4, 5, 6, 1, 2, 3]
                if largest >= 1:
                    return [1, 2, 3]
                else:
                    return []
            if dimension == 1:
                if largest >= 131072:
                    return [18, 15, 17, 12, 14, 16, 9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 65536:
                    return [15, 17, 12, 14, 16, 9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 32768:
                    return [12, 14, 16, 9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 16384:
                    return [9, 11, 13, 6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 8192:
                    return [6, 8, 10, 3, 5, 7, 2, 4, 1]
                if largest >= 4096:
                    return [3, 5, 7, 2, 4, 1]
                if largest >= 2048:
                    return [2, 4, 1]
                if largest >= 1024:
                    return [1]
                else:
                    return []
        def confirm_flavor(flavor, box):
            for number in flavor:
                if number in box:
                    return number
            return None
        first_largest = first_select_box[0][0]
        second_largest = second_select_box[0][1]
        recommend_number_one = good_flavor(first_largest, 0)#返回剩余量对应的推荐箱子号
        recommend_number_two = good_flavor(second_largest, 1)#返回剩余量对应的推荐箱子号
#返回可能交换的flavor序列

        while(True):

            first_box_push = confirm_flavor(recommend_number_one, second_box_to)#准备交换的箱子是否存在该flavor
            second_box_push = confirm_flavor(recommend_number_two, first_box_to)#判断箱子里是否有准备交换箱子的存在

            if(first_box_push == None or second_box_push == None):
                return None, None

            if(first_box_push == second_box_push):
                if(len(recommend_number_one) > len(recommend_number_two)):
                    del(recommend_number_one[recommend_number_one.index(first_box_push)])

                    if(len(recommend_number_one) == 0):
                        return None, None
                    else:
                        continue
                else:
                    del(recommend_number_two[recommend_number_two.index(second_box_push)])

                    if(len(recommend_number_two) == 0):
                        return None, None
                    else:
                        continue  


            first_surplus_extracted = summu_cpu_men(first_select_box[0], flavor_type[second_box_push])#是否有bug
            second_surplus_extracted = summu_cpu_men(second_select_box[0], flavor_type[first_box_push])

            if (calculate_bigger(first_surplus_extracted, flavor_type[first_box_push]) and calculate_bigger(second_surplus_extracted, flavor_type[second_box_push])):
  
                return second_box_push, first_box_push #返回要和另外一个交换的flavor序号
            elif(calculate_bigger(first_surplus_extracted, flavor_type[first_box_push]) == False): 
                
                del(recommend_number_one[recommend_number_one.index(first_box_push)])
#                print("recommend_number_one")
#                print(recommend_number_one)
                if(len(recommend_number_one) == 0):
                    return None, None
            elif(calculate_bigger(second_surplus_extracted, flavor_type[second_box_push]) == False):
            
                del(recommend_number_two[recommend_number_two.index(second_box_push)])
#                print("recommend_number_two")
#                print(recommend_number_two)
                if(len(recommend_number_two) == 0):
                    return None, None
        
    score=evaluation_score(flavor_list[0],flavor_list[1], flavor_type, box_type)
    print("score,not addition box_number") 
    print(score) 
    all_list = sort_box_vector(flavor_list, flavor_type, box_type)
    #返回值为串中箱子对应的的剩余数和每个箱子flavor序列，以及箱子规则
    last_list=[]
    for i in range(0,4):
        last_list = addition_flavor(all_list, maybe_flavor, box_type, flavor_type)
        #last_list 返回值为串中箱子对应的的剩余数和每个箱子flavor序列，以及箱子规则 
        tmp=[]
        flavor_box1=[]
        flavor_box2=[]
        for i in range(len(last_list)):
            flavor_box1.append(last_list[i][1])
            flavor_box2.append(last_list[i][2])
        tmp.append(flavor_box1)
        tmp.append(flavor_box2)
        last_list=tmp
        all_list = sort_box_vector(last_list, flavor_type, box_type)
    flavor_box1=[]
    flavor_box2=[]
    for i in range(len(all_list)):
        flavor_box1.append(all_list[i][1])
        flavor_box2.append(all_list[i][2])
    score=evaluation_score(flavor_box1,flavor_box2, flavor_type, box_type)
    print("score,addiction cheat 4 times ") 
    print(score)
    #二轮交换
    for i in range(times_change):#last_list 箱子的装箱情况及对应序列，二维
        #print("i:",i)
        surplus_boxnumber=calculate_surplus(last_list, flavor_type, box_type) #计算箱子剩余量，加索引
        first_select_box, second_select_box = selection_best_change_box(surplus_boxnumber)
        #返回经过轮盘选择以后要交换余量的箱子及索引号

        flavor_exchange_second, flavor_exchange_first = exchange_flavor_twobox(first_select_box, last_list[0][first_select_box[-1]], 
                                      second_select_box, last_list[0][second_select_box[-1]], 
                                      flavor_type)
        if(flavor_exchange_second == None or flavor_exchange_first == None):
            continue
        del(last_list[0][first_select_box[-1]][last_list[0][first_select_box[-1]].index(flavor_exchange_second)])
        last_list[0][first_select_box[-1]].append(flavor_exchange_first)
        del(last_list[0][second_select_box[-1]][last_list[0][second_select_box[-1]].index(flavor_exchange_first)])
        last_list[0][second_select_box[-1]].append(flavor_exchange_second)
        #        print(last_list)
        all_list = sort_box_vector(last_list, flavor_type, box_type)
        score_list = addition_flavor(all_list, maybe_flavor, box_type, flavor_type)
        tmp=[]
        flavor_box1=[]
        flavor_box2=[]
        for i in range(len(score_list)):
            flavor_box1.append(score_list[i][1])
            flavor_box2.append(score_list[i][2])
        tmp.append(flavor_box1)
        tmp.append(flavor_box2)
        last_list=tmp
        
        surpluses = calculate_surplus(last_list,flavor_type, box_type)
        print("the box surpluse:",surpluses)
        print("the score you get is:")
        score=evaluation_score(last_list[0],last_list[1], flavor_type, box_type)
        print("maybe_flavor:", maybe_flavor)
        #print("score,addiction other box_number") 
        print(score)
    return last_list
def generation_result(flavor_names,flavor_list_box):
    flavor_prediction={}
    print('flavor_names',flavor_names)
    print('flavor_list_box',flavor_list_box)
    for single_box_allocate in flavor_list_box:
        set_single_server=set(single_box_allocate)
        for flavor in set_single_server:
            if flavor_prediction.has_key('flavor'+str(flavor)):
                flavor_prediction['flavor'+str(flavor)]+=single_box_allocate.count(flavor)
            else:
                flavor_prediction['flavor'+str(flavor)]=single_box_allocate.count(flavor)
    print ('flavor_prediction',flavor_prediction)

    res=''
    sum_of_flavors=0
    for name in flavor_names:
        if flavor_prediction.has_key(name)==False:
            flavor_prediction[name]=0
        sum_of_flavors+=flavor_prediction[name]
        res+=name+' '+str(flavor_prediction[name])+'\n'
    res=str(sum_of_flavors)+'\n'+res
    return res

def pack_flavor(flavor_NE,flavor_predict_list,flavor_type_name,server_type,box_name):
    
    prediction_dict={}#预测的flavor的字典表示
    flavor_type={}
    for name in flavor_NE:
        prediction_dict[int(name[6:])] = flavor_predict_list[name]#prediction_dict={1:2,2:3,...}
        flavor_type[int(name[6:])] = flavor_type_name[name]#统一表示为数字和size flavor_type={1: [1, 1024], 2: [1, 2048],..}
    box_type=[[box_single[0],box_single[1]*1024] for box_single in server_type]#三种箱子大小box_type= [[56, 128], [84, 256], [112, 192]]
#     print ('box_type:', box_type)
    start_time = time.clock()
    run_time =50
    number_population = 150
    times_change=1000

    flavor_type = {1: [1, 1024], 2: [1, 2048], 3: [1, 4096], 4: [2, 2048], 5: [2, 4096], 6: [2, 8192],7: [4, 4096], 8: [4, 8192], 9: [4, 16384], 10: [8, 8192], 11: [8, 16384],12: [8, 32768], 13: [16, 16384], 14: [16, 32768], 15: [16, 65536], 16: [32, 32768],17: [32, 65536], 18: [32, 131072]}
    allocate_result= second_pack(prediction_dict, box_type, flavor_type, number_population, start_time,run_time, good_boy=number_population /7)
    print("allocate_result:",allocate_result)#返回分数，每个对应flavor序列，以及箱子序列（可能有多个选择）
    good_list = selection_good_list(allocate_result,box_type, flavor_type) #挑选allocate_result中空间比较优的序列
    print("good_list:",good_list)#返回分数，每个对应flavor序列，以及箱子序列 （'good_list:', [[9227114, ([[11, 7, 4, 8, 13, 11, ]...)
    flavor_box_list, surpluse_size, good_best_flavor = choose_best_list(good_list, flavor_type, box_type)
    #best_list 最高分数的flavor序列。包含分数，装箱序列以及箱子规则序列
    #返回每个箱子flavor序列和箱子序列
    #last_flavor_list装箱情况及对应的箱子类型
    last_flavor_list=flavor_box_list
    print(last_flavor_list)
    last_box=optimize_box(last_flavor_list, flavor_type, box_type, prediction_dict,times_change)
    
    print(last_box)
    box_list_last=last_box[1]#箱子序列
    flavor_list_last=last_box[0]#装箱情况,flavor序列
    res=''
    server_count=[box_list_last.count(0),box_list_last.count(1),box_list_last.count(2)]
    for index,count in enumerate(server_count):
        if count==0:
            continue
        res+=box_name[index]+' '+str(count)+'\n'
        server_i_count=1
        for index_alloc,alloc in enumerate(flavor_list_last):
            if box_list_last[index_alloc]!=index:
                continue
            res+=box_name[index]+'-'+str(server_i_count)+' '
            flavor_set = set(alloc)
            for single in flavor_set:
                res+='flavor'+str(single)+' '+str(alloc.count(single))+' '
            res+='\n'
            server_i_count+=1
        res+='\n'


    # print("res_alloc",res_alloc)
    res_predict=generation_result(flavor_NE,last_box[0])
    res=res_predict+'\n'+res
    
    return res

def test():
    flavor_NE = [  'flavor3', 'flavor4', 'flavor5', 'flavor6', 'flavor7', 'flavor8','flavor9', 'flavor10', 'flavor11','flavor12', 'flavor13','flavor14','flavor15','flavor16','flavor17','flavor18']
   #所需预测的flavor,'flavor2': 44,'flavor2',
    flavor_prediction = {  'flavor3': 12, 'flavor4': 26, 'flavor5': 10, 'flavor6': 0,
                         'flavor7': 50, 'flavor8': 10, 'flavor9': 22, 'flavor10': 3,'flavor11': 36, 'flavor12': 7, 'flavor13': 9,
                         'flavor14': 8, 'flavor15': 15,'flavor16':4,'flavor17':5,'flavor18':0}
#     flavor_prediction = { 'flavor2': 0, 'flavor3': 0, 'flavor4': 10, 'flavor5': 0, 'flavor6': 10,
#                          'flavor7': 5, 'flavor8': 10, 'flavor9': 5, 'flavor10': 15,'flavor11': 0, 'flavor12': 12, 'flavor13': 5,
#                          'flavor14': 8, 'flavor15': 10,'flavor16':0,'flavor17':0,'flavor18':5}
    #所需预测的flavor的个数
    flavor_type = {'flavor1': [1, 1024], 'flavor2': [1, 2048], 'flavor3': [1, 4096], 'flavor4': [2, 2048],
                   'flavor5': [2, 4096], 'flavor6': [2, 8192], 'flavor7': [4, 4096], 'flavor8': [4, 8192],
                   'flavor9': [4, 16384], 'flavor10': [8, 8192], 'flavor11': [8, 16384], 'flavor12': [8, 32768],
                   'flavor13': [16, 16384], 'flavor14': [16, 32768], 'flavor15': [16, 65536],'flavor16':[32, 32768], 'flavor17':[32, 65536], 'flavor18':[32, 131072]}
    #所需预测的flavor的尺寸大小
    box_type= [[56, 128], [84, 256], [112, 192]]
    #box_type = [[56, 128], [84, 256]]
    box_name=['General','Large-Memory','High-Performance']
    #box_name=['General','Large-Memory']
    result = pack_flavor(flavor_NE, flavor_prediction, flavor_type, box_type,box_name)

    print result

if __name__ == '__main__':
    test()
