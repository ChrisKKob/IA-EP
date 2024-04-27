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

#print(amostraFinal)

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

numeroNeuroniosEscondidos = 5

numeroAmostraAtual = 0

bias = random.uniform(0.00000001, 1.0)

#neuronios da camada escondida
class NeuronioEscondido:

    def __init__(self):
        self.pesos = np.random.random(120)

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
        return sigmoide(self.somaPesos() + bias)


neuroniosEscondidos = [NeuronioEscondido() for _ in range(numeroNeuroniosEscondidos)]

#neuronio da camada de saida
class NeuronioSaida:

    valorSaida = 0.0

    def __init__(self):
        self.pesos = np.random.random(numeroNeuroniosEscondidos)

    def somaPesos(self):
        somatorio = 0.0
        for i in range(0, numeroNeuroniosEscondidos - 1):
            somatorio += neuroniosEscondidos[i].ativacao() * self.pesos[i]
        return somatorio
    
    def ativacao(self):
        return sigmoide(self.somaPesos() + bias)

neuronioSaida = [NeuronioSaida() for _ in range(26)]

print(neuronioSaida[0].somaPesos())


mapeamentoRotulo = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
              'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19,
              'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}

####################################################################

#
####################################################################