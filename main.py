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
        self.pesos = np.random.rand(120)

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


class NeuronioSaida:

    valorSaida = float

    def __init__(self):
        self.pesos = np.random.rand(numeroNeuroniosEscondidos)

    def somaPesos(self):
        somatorio = 0.0
        for i in range(0, numeroNeuroniosEscondidos - 1):
            somatorio += neuroniosEscondidos[i].ativacao() * self.pesos[i]
        return somatorio

neuronioSaida = [NeuronioSaida() for _ in range(26)]

print(neuronioSaida[0].somaPesos())


####################################################################