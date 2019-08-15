import random as rd
import numpy as np


class Gen_Selection:

    def __init__(self):
        pass
    
    def Calc_Fitness(self,score,E_Consumption,F_Constant):
        F_Cons = F_Constant; Fitness_Value = []
        E_Cons = list(); E_Cons = E_Consumption
        StanScore = (score-np.mean(score,axis=0))/np.std(score,axis=0) #Making Stan
        for i in range(len(StanScore)):
            Fitness_Value.append(F_Cons*(StanScore[i]/E_Cons[i]))
            
        return Fitness_Value
    
    def RWCalc_And_Selection(self,Fitness_Value,Gen_List):
        Probability_Storage = []
        Fitness_Value = Fitness_Value - min(Fitness_Value)
        sum_Fitness = sum(Fitness_Value)
        for i in range(len(Gen_List)):
            Probability_Storage.append(abs(Fitness_Value[i]/sum_Fitness))
            
        Gen1 = []; Gen2 = []
        cnt = 0
        m_list = []; fm_list =[]
        for i in range(len(Gen_List)):
            m_list.append(i)
        for i in range(len(Gen_List)):
            fm_list.append(np.random.choice(m_list,p=Probability_Storage))
        while(cnt<8):
            Gen1.append(Gen_List[fm_list[cnt]])
            Gen2.append(Gen_List[fm_list[cnt+1]])
            cnt+=2
            
        return Gen1,Gen2


class Chromosomes_Offset:

    def __init__(self):
        pass

    def initGen(self,n):
        gen_list = []
        for i in range(n):
            gen_Master = str()
            for _ in range(30):
                gen_Master+=str(rd.randint(0,1))
            gen_Master += '111'
            gen_Slave = str(); converter = lambda x: [x[3 * i:3 * i + 3] for i in range(11)]
            Gen_Master = converter(gen_Master)
            for i in range(11):
                if((Gen_Master[i]=='110')or(Gen_Master[i]=='011')):
                    if(Gen_Master[i]=='110'):
                        gen_Slave+='ff'
                    else:
                        gen_Slave+='00'
                else:
                    RandInt = rd.randint(1,244)
                    RandHex = format(RandInt, 'x')
                    if(len(RandHex)<2):
                        RandHex = '0'+RandHex
                        gen_Slave += RandHex
                    else:
                        gen_Slave+= RandHex
            gen_list.append([gen_Master,gen_Slave])

        return gen_list

                
class Mutation:

    def __init__(self):
        pass

    def genToList(self,a,gen):
        if(a==0): #Master_Gene
            converter = lambda x: [x[3 * i:3 * i + 3] for i in range(11)]
            return converter(gen) #converter converts gen to list
        else: #Slave_Gene
            converter =  lambda x: [x[2 * i:2 * i + 2] for i in range(11)]
            return converter(gen)

    def Mutation_Rates_Calculation(self,score):
        alpha = 15; mutate_rate= float(alpha/score)
        return mutate_rate
    
    def Mutation(self,mixed_gene1_master,mixed_gene1_slave,mixed_gene2_master,mixed_gene2_slave,mutate_rate):
        mixed_gene1_master = self.genToList(0,mixed_gene1_master)
        mixed_gene2_master = self.genToList(0,mixed_gene2_master)
        mixed_gene1_slave = self.genToList(1,mixed_gene1_slave)
        mixed_gene2_slave = self.genToList(1,mixed_gene2_slave)
        mixed_gene1_m = []; mixed_gene2_m = []; mixed_gene1_s = []; mixed_gene2_s = []
        for i in range(11):
            if(i==10):
                mixed_gene1_m.append('111');mixed_gene2_m.append('111')
                MP1 = str(format(rd.randint(1,254),'x'))
                if(len(MP1)<2):
                    MP1 = '0'+MP1;mixed_gene1_s.append(MP1)
                else:
                    mixed_gene1_s.append(MP1)
                MP2 = str(format(rd.randint(1,254),'x'))
                if(len(MP2)<2):
                    MP2 = '0'+MP2
                    mixed_gene2_s.append(MP2)
                else:
                    mixed_gene2_s.append(MP2)
                break

            elif rd.random() <= mutate_rate:
                mixed_gene1_m.append("{0:b}".format(rd.randint(0, 7)).zfill(3))
                mixed_gene2_m.append("{0:b}".format(rd.randint(0, 7)).zfill(3))
                if (((mixed_gene1_m[i]=='110')or(mixed_gene1_m[i]=='011'))==1):
                    if(mixed_gene1_m[i]=='011'):
                        if(mixed_gene1_slave[i]!='00'):
                            mixed_gene1_s.append('00')
                        else:
                            mixed_gene1_s.append(mixed_gene1_slave[i])
                    else:
                        if(mixed_gene1_slave[i]!='ff'):
                            mixed_gene1_s.append('ff')
                        else:
                            mixed_gene1_s.append(mixed_gene1_slave[i])
                else:
                    MP1 = str(format(rd.randint(1,254),'x'))
                    if(len(MP1)<2):
                        MP1 = '0'+MP1;mixed_gene1_s.append(MP1)
                    else:
                        mixed_gene1_s.append(MP1)

                if(((mixed_gene2_m[i]=='110')or(mixed_gene2_m[i]=='011'))==1):
                    if(mixed_gene2_m[i]=='011'):
                        if(mixed_gene2_slave[i]!='00'):
                            mixed_gene2_s.append('00')
                        else:
                            mixed_gene2_s.append(mixed_gene2_slave[i])
                    else:
                        if(mixed_gene2_slave[i]!='ff'):
                            mixed_gene2_s.append('ff')
                        else:
                            mixed_gene2_s.append(mixed_gene2_slave[i])
                else:
                    MP2 = str(format(rd.randint(1,254),'x'))
                    if(len(MP2)<2):
                        MP2 = '0'+MP2
                        mixed_gene2_s.append(MP2)
                    else:
                        mixed_gene2_s.append(MP2)
            else:
                mixed_gene1_m.append(mixed_gene1_master[i])
                mixed_gene1_s.append(mixed_gene1_slave[i])
                mixed_gene2_m.append(mixed_gene2_master[i])
                mixed_gene2_s.append(mixed_gene2_slave[i])

        return mixed_gene1_m, mixed_gene2_m, mixed_gene1_s, mixed_gene2_s


class CrossOver:
    
    def __init__(self):
        pass

    def genToList(self,a,gen):
        if(a==0): #Master_Gene
            converter = lambda x: [x[3 * i:3 * i + 3] for i in range(11)]
            return converter(gen) #converter converts gen to list
        else: #Slave_Gene
            converter =  lambda x: [x[2 * i:2 * i + 2] for i in range(11)]
            return converter(gen)
    
    def shuffle_Master(self, gen1_master, gen2_master):
        Gen = list();Gen = [[0] * 11 for i in range(2)]
        Gen[0] = self.genToList(0,gen1_master)
        Gen[1] = self.genToList(0,gen2_master)
        gen_shuffled1 = list();gen_shuffled2 = list()
        Inc = 1
        for i in range(11):
            if (Inc == 1):
                gen_shuffled1 += Gen[0][i]
                if (((Gen[0][i] == '011') or (Gen[0][i] == '110'))==1):
                    Inc -= 1
            elif(Inc==0):
                gen_shuffled1 += Gen[1][i]
                if (((Gen[1][i] == '011') or (Gen[1][i] == '110'))==1):
                    Inc += 1
        Inc = 1
        for i in range(11):
            if (Inc==1):
                gen_shuffled2 += Gen[1][i]
                if (((Gen[1][i] == '011') or (Gen[1][i] == '110'))==1):
                    Inc -= 1
            elif(Inc==0):
                gen_shuffled2 += Gen[0][i]
                if (((Gen[0][i] == '011') or (Gen[0][i] == '110'))==1):
                    Inc += 1
        gen_shuffled1_master = ''.join(gen_shuffled1)
        gen_shuffled2_master = ''.join(gen_shuffled2)

        return gen_shuffled1_master, gen_shuffled2_master

    def shuffle_Slave(self,gen1_slave,gen2_slave):
        Gen = list();Gen = [[0] * 11 for i in range(2)]
        Gen[0] = self.genToList(1,gen1_slave)
        Gen[1] = self.genToList(1,gen2_slave)
        gen_shuffled1 = list();gen_shuffled2 = list()
        Inc = 1
        for i in range(11):
            if (Inc == 1):
                gen_shuffled1 += Gen[0][i]
                if (((Gen[0][i] == '00') or (Gen[0][i] == 'ff'))==1):
                    Inc -= 1
            elif(Inc==0):
                gen_shuffled1 += Gen[1][i]
                if (((Gen[1][i] == '00') or (Gen[1][i] == 'ff'))==1):
                    Inc += 1
        Inc = 1
        for i in range(11):
            if (Inc == 1):
                gen_shuffled2 += Gen[1][i]
                if (((Gen[1][i] == '00') or (Gen[1][i] == 'ff'))==1):
                    Inc -= 1
            elif(Inc==0):
                gen_shuffled2 += Gen[0][i]
                if (((Gen[0][i] == '00') or (Gen[0][i] == 'ff'))==1):
                    Inc += 1
        gen_shuffled1_slave = ''.join(gen_shuffled1)
        gen_shuffled2_slave = ''.join(gen_shuffled2)

        return gen_shuffled1_slave , gen_shuffled2_slave
