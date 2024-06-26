import numpy as np
import random
from sklearn.model_selection import train_test_split

#importacao dos dados
####################################################################

#funcao para remover espacos do arquivo X.txt
def removerEspaco(arquivo):
    with open(arquivo, 'r') as arq:
        dados = arq.read()
        dadosOrganizados = dados.replace(' ', '')
    return(dadosOrganizados)
    
nomeArquivo = 'X.txt'
entradaProcessada = removerEspaco(nomeArquivo)

#configuração para imprimir todo o array
np.set_printoptions(threshold=np.inf)

#extraindo os dados das amostras do arquivo X
amostraAux = np.fromstring(entradaProcessada, sep=',')

#remodelando os dados que ficaram em 1 dimensao para uma matriz
amostra = np.reshape(amostraAux, (1326, 120))

amostra_treino, resto = train_test_split(amostra, train_size=0.7, random_state=42)

amostra_validacao, amostra_teste = train_test_split(resto, test_size=0.5, random_state=42)


#carregamento dos dados do rotulo dos dados
rotulo = np.loadtxt('Y_letra.txt',dtype=str)

#print(rotulo)

####################################################################
#
# variaveis definidas: amostra_treino, amostra_validacao, amostra_teste, rotulo
#


#definicao da base da arquitetura da mlp
####################################################################

#funcao de ativacao sigmoide
def sigmoide(t):
    return 1/(1+np.exp(-t))

#derivada da funcao sigmoide
def derivadaSigmoide(p):
    return p * (1 - p)

#numero de neuronios na camada escondida
numeroNeuroniosEscondidos = 5

numeroAmostraAtual = 0

taxaDeAprendizado = 0.6

#neuronios da camada escondida
class NeuronioEscondido:

    def __init__(self):
        self.pesos = np.random.random(120)
        self.valorSaida = 0.0
        self.termoErro = 0.0
        self.bias = random.uniform(0.000001, 1.0)

    def printPesos(self):
        print(self.pesos)

    #soma todos os pesos multiplicados pela entrada do neuronio
    def somaPesos(self):
        somatorio = 0.0
        for i in range(0, 120):
            somatorio += amostra_treino[numeroAmostraAtual][i] * self.pesos[i]
        return somatorio
    
    #retorna o resultado da funcao de ativacao
    def ativacao(self):
        self.valorSaida = sigmoide(self.somaPesos() + self.bias)

    def salvarTermoErro(self, valorErro):
        self.termoErro = valorErro

    def atualizarPesos(self):
        deltaPeso = 0.0
        for i in range(0, 120):
            deltaPeso = self.pesos[i] * taxaDeAprendizado * self.termoErro
            self.pesos[i] += deltaPeso
        self.bias = self.bias + taxaDeAprendizado * self.termoErro

    


neuroniosEscondidos = [NeuronioEscondido() for _ in range(numeroNeuroniosEscondidos)]

#neuronio da camada de saida
class NeuronioSaida:

    def __init__(self):
        self.pesos = np.random.random(numeroNeuroniosEscondidos)
        self.valorSaida = 0.0
        self.termoErro = 0.0
        self.bias = random.uniform(0.000001, 1.0)

    def somaPesos(self):
        somatorio = 0.0
        for i in range(0, numeroNeuroniosEscondidos):
            somatorio += neuroniosEscondidos[i].valorSaida * self.pesos[i]
        return somatorio
    
    def ativacao(self):
        self.valorSaida = sigmoide(self.somaPesos() + self.bias)

    def salvarTermoErro(self, valorErro):
        self.termoErro = valorErro

    def atualizarPesos(self):
        deltaPeso = 0.0
        for i in range(0, numeroNeuroniosEscondidos):
            deltaPeso = self.pesos[i] * self.termoErro * taxaDeAprendizado
            self.pesos[i] += deltaPeso
        self.bias = self.bias + self.termoErro * taxaDeAprendizado




neuroniosSaida = [NeuronioSaida() for _ in range(26)]

mapeamentoRotulo = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
              'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
              'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}



####################################################################

#Execução da arquitetura
####################################################################

numeroEpoca = 0
limiteEpoca = 42
acerto = 0

while True:
    #condiçoes de parada
    if(numeroEpoca >= limiteEpoca):
        break

    #Feedforward
    for i in range(0, numeroNeuroniosEscondidos):
        neuroniosEscondidos[i].ativacao()

    for i in range(0, 26):
        neuroniosSaida[i].ativacao()
        #print(neuroniosSaida[i].valorSaida)

    erro = False

    for i in range(0, 26):
        if(i == mapeamentoRotulo[rotulo[numeroAmostraAtual]]):
            if(neuroniosSaida[i].valorSaida <= 1): erro = True
        elif(neuroniosSaida[i].valorSaida >= 0): erro = True

    #Backpropagation
    #Calculando e salvando o termo de erro da camada de saida
    somatorioTermoErroOculto = 0.0
    if(erro):
        for i in range(0, 26):
            if(i == mapeamentoRotulo[rotulo[numeroAmostraAtual]]):
                neuroniosSaida[i].salvarTermoErro(1 - neuroniosSaida[i].valorSaida * derivadaSigmoide(neuroniosSaida[i].valorSaida))
            else: neuroniosSaida[i].salvarTermoErro(0 - neuroniosSaida[i].valorSaida * derivadaSigmoide(neuroniosSaida[i].valorSaida)) 
        
    #Calculo do termo de erro para os neuronios da camada oculta        
        for i in range(0, numeroNeuroniosEscondidos):
            for j in range(0, 26):
                somatorioTermoErroOculto += neuroniosSaida[j].termoErro * neuroniosSaida[j].pesos[i]
            neuroniosEscondidos[i].salvarTermoErro(somatorioTermoErroOculto * derivadaSigmoide(neuroniosEscondidos[i].valorSaida))

        if(numeroEpoca == 42): print(somatorioTermoErroOculto)

    #Atualizando pesos
        for i in range(0, 26):
            neuroniosSaida[i].atualizarPesos()

        for i in range(0, numeroNeuroniosEscondidos):
            neuroniosEscondidos[i].atualizarPesos()
    else: acerto += 1

    if(acerto == 928):
        break

    #parametros do loop para passar nova amostra
    numeroAmostraAtual += 1

    print("Epoca "+ str(numeroEpoca + 1)+"/ numero de acertos:"+str(acerto)+"/ Numero amostra:" +str(numeroAmostraAtual))

    if(numeroAmostraAtual >= len(amostra_treino)):
        numeroEpoca += 1
        numeroAmostraAtual = 0
        acerto = 0
        np.random.shuffle(amostra_treino)



