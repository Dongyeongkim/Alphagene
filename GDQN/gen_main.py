import Genetic
import Model_Converter
import numpy as np
import os

# Define the Function that will be used 
class GenMain:
    def __init__(self):
        self.First_Gen = Genetic.Chromosomes_Offset()
        self.Gen_S = Genetic.Gen_Selection()
        self.Gen_C = Genetic.CrossOver()
        self.Gen_M = Genetic.Mutation()
        self.PathG = './CrS/'
        self.PathS = './Score/score.txt'
        self.PathE ='./E_Cons/E_Cons.txt'

    def Get_Gene(self,path):
        Gen_List = []
        if (len(os.listdir(path))>8):
            Offset = os.listdir(path);print(Offset)
            for gene in Offset:
                if(gene=='.DS_Store'):
                    pass
                else:
                    f = open(path+gene,'r')
                    Gen = f.readlines()
                    M_Gen = Gen[0]; S_Gen = Gen[1]
                    Gen_List.append([M_Gen,S_Gen]); f.close()
        else:
            Gen_List = self.First_Gen.initGen(8)
            for i in range(len(Gen_List)):
                Mgen=Gen_List[i][0];print(Gen_List[i][0])
                f = open(path+Mgen+'.txt','w')
                Gene = '\n'.join(Gen_List[i])
                f.write(Gene)
                f.close()
            return Gen_List
    
    def Get_Score(self,path):
        if (os.path.isfile(path)==True):
            f = open(path,'r'); data = f.read(); data = data.split()
            f.close()
        else:
            data = [1,2,3,4,5,6,7,8]
        return data

    def Get_ECons(self,path):
        if os.path.exists(path):
            f = open(path,'r')
            E_Cons = f.read(); E_Cons = E_Cons.split()
            f.close()
        else:
            E_Cons_E = [1,2,3,4,5,6,7,8]
            E_Cons = E_Cons_E

        return E_Cons
        
    def main(self):
        Gen_List = self.Get_Gene(self.PathG)
        score = self.Get_Score(self.PathS)
        E_Cons = self.Get_ECons(self.PathE)
        F_Cons = 10

        Fitness_Value = self.Gen_S.Calc_Fitness(score,E_Cons,F_Cons)
        gen1, gen2 = self.Gen_S.RWCalc_And_Selection(Fitness_Value,Gen_List)

        Fin_Gen = []
        mutate_rate = self.Gen_M.Mutation_Rates_Calculation(100)
        ##Crossover Gene
        for i in range(len(gen1)):
            MMG1, MMG2 = self.Gen_C.shuffle_Master(gen1[i][0],gen2[i][0])
            MSG1, MSG2 = self.Gen_C.shuffle_Slave(gen1[i][1],gen2[i][1])
            
            ##Mutating Gene
            FMG1,FMG2,FSG1,FSG2= self.Gen_M.Mutation(MMG1,MSG1,MMG2,MSG2,mutate_rate)
            FMG1 = ''.join(FMG1); FMG2 = ''.join(FMG2)
            FSG1 = ''.join(FSG1); FSG2 = ''.join(FSG2)
            Fin_Gen.append([FMG1,FSG1]); Fin_Gen.append([FMG2,FSG2])

        ##Save Gene

        for i in range(len(Fin_Gen)):
            print(Fin_Gen[i])

        Mgen = []; Sgen = []

        for i in range(len(Fin_Gen)):
            Mgen.append(Fin_Gen[i][0])
            Sgen.append(Fin_Gen[i][1])

        model = []
        for i in range(len(Mgen)):
            model.append(Model_Converter.GeneticModel(Mgen[i],Sgen[i]))

        return model

