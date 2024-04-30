import numpy as np
import random

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

#print(amostra)

#carregamento dos dados do rotulo dos dados
rotulo = np.loadtxt('Y_letra.txt',dtype=str)

#print(rotulo)

####################################################################
#
# variaveis definidas: amostra, rotulo
#


#definicao da base da arquitetura da mlp
####################################################################

#funcao de ativacao sigmoide
def sigmoide(t):
    return 1/(1+np.exp(-t))

#derivada da funcao sigmoide
def derivadaSigmoide(p):
    return sigmoide(p) * (1 - sigmoide(p))

#numero de neuronios na camada escondida
numeroNeuroniosEscondidos = 5

numeroAmostraAtual = 0

bias = random.uniform(0.000001, 1.0)

taxaDeAprendizado = 0.4

#neuronios da camada escondida
class NeuronioEscondido:

    def __init__(self):
        self.pesos = np.random.random(120)
        self.valorSaida = 0.0

    def printPesos(self):
        print(self.pesos)

    #soma todos os pesos multiplicados pela entrada do neuronio
    def somaPesos(self):
        somatorio = 0.0
        for i in range(0, 119):
            somatorio += amostra[numeroAmostraAtual][i] * self.pesos[i]
        return somatorio
    
    #retorna o resultado da funcao de ativacao
    def ativacao(self):
        self.valorSaida = sigmoide(self.somaPesos() + bias)

    def getSaida(self):
        return self.valorSaida


neuroniosEscondidos = [NeuronioEscondido() for _ in range(numeroNeuroniosEscondidos)]


#neuronio da camada de saida
class NeuronioSaida:

    def __init__(self):
        self.pesos = np.random.random(numeroNeuroniosEscondidos)
        self.valorSaida = 0.0

    def somaPesos(self):
        somatorio = 0.0
        for i in range(0, numeroNeuroniosEscondidos):
            somatorio += neuroniosEscondidos[i].getSaida() * self.pesos[i]
        return somatorio
    
    def ativacao(self):
        self.valorSaida = sigmoide(self.somaPesos() + bias)


neuroniosSaida = [NeuronioSaida() for _ in range(26)]

mapeamentoRotulo = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
              'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
              'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

####################################################################

#
####################################################################

numeroEpoca = 0

while True:
    #condiçoes de parada
    if(numeroEpoca >= 1):
        break

    #Feedforward
    for i in range(0, numeroNeuroniosEscondidos - 1):
        neuroniosEscondidos[i].ativacao()

    for i in range(0, 25):
        neuroniosSaida[i].ativacao()
        print(neuroniosSaida[i].valorSaida)

    numeroEpoca += 1


