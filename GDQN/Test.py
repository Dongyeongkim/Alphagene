import Genetic
import Model_Converter
import numpy as np

# Define the Function that will be used 

First_Gen = Genetic.Chromosomes_Offset()
Gen_S = Genetic.Gen_Selection()
Gen_C = Genetic.CrossOver()
Gen_M = Genetic.Mutation()


##Set Offsets

Gen_List = First_Gen.initGen(8)

##Select Gene

score = [0,1,2,3,4,5,6,7]
E_Cons = [2.1,3.4,0.3,0.9,1.1,1.4,1.3,2.7]
F_Cons = 10

Fitness_Value = Gen_S.Calc_Fitness(score,E_Cons,F_Cons)
gen1, gen2 = Gen_S.RWCalc_And_Selection(Fitness_Value,Gen_List)

Fin_Gen = []
mutate_rate = Gen_M.Mutation_Rates_Calculation(100)
##Crossover Gene
for i in range(len(gen1)):
    MMG1, MMG2 = Gen_C.shuffle_Master(gen1[i][0],gen2[i][0])
    MSG1, MSG2 = Gen_C.shuffle_Slave(gen1[i][1],gen2[i][1])
    
    ##Mutating Gene
    FMG1,FMG2,FSG1,FSG2=Gen_M.Mutation(MMG1,MSG1,MMG2,MSG2,mutate_rate)
    FMG1 = ''.join(FMG1); FMG2 = ''.join(FMG2)
    FSG1 = ''.join(FSG1); FSG2 = ''.join(FSG2)
    Fin_Gen.append([FMG1,FSG1]); Fin_Gen.append([FMG2,FSG2])

##Save Gene

for i in range(len(Fin_Gen)):
    print(Fin_Gen[i])

Mgen = Fin_Gen[0][0]
Sgen = Fin_Gen[0][1]

Mgen = []; Sgen = []

for i in range(len(Fin_Gen)):
    Mgen.append(Fin_Gen[i][0])
    Sgen.append(Fin_Gen[i][1])

model = []
model.append(Model_Converter.GeneticModel(Mgen[0],Sgen[0]))
model.append(Model_Converter.GeneticModel(Mgen[1],Sgen[1]))
model.append(Model_Converter.GeneticModel(Mgen[2],Sgen[2]))
model.append(Model_Converter.GeneticModel(Mgen[3],Sgen[3]))
model.append(Model_Converter.GeneticModel(Mgen[4],Sgen[4]))
model.append(Model_Converter.GeneticModel(Mgen[5],Sgen[5]))
model.append(Model_Converter.GeneticModel(Mgen[6],Sgen[6]))
model.append(Model_Converter.GeneticModel(Mgen[7],Sgen[7]))

for i in range(len(model)):
    print(model[i])




