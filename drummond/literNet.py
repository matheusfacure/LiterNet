# -*- coding: latin-1 -*-
from __future__ import absolute_import, division


import tflearn
from tflearn.data_utils import *

path = "drummond_copil" # arquivo de texto 
save_file = 'LiterNet_cpk' # arquivo para salvar a rede neural


##############################################################
######## Hiper-parâmetros da rede neural recursiva ###########
##############################################################

# Para modelar a sequência de caracteres, vamos utilizar uma
# rede neural recursiva. Nós vamos apresentar à ela os
# caracteres em sequência e pedir que ela preveja o próximo
# caracteres. A RNR aprenderá a dinâmica da sequência
# de caracteres, isto é, como eles consumam aparecer
# sucessivamente na sequência.

# Como nossos recursos computacionais não são muito altos,
# teremos que limitar o tamanho da sequência de caracteres
# para um tamanho máximo maxlen. Assim, ao prever o próximo
# caractere, a RNR não poderá utilizar informação de palavras
# que aparecem muito atrás no texto. Isso impedirá que ela
# consiga aprender contextos mais amplos. O limite da
# sequência será definido por maxlen

# o resto dos hiper-parâmetros são bastante simples:
# drop_porb é a probabilidade de manter os dados
# durante dropout, lr é a taxa de aprendizado,
# batch_size é o tamanho do mini lote e hl1, hl2 e
# hl3 são os tamanhos das camadas ocultas 1, 2 e 3
# respectivamente.

# Vamos gerar sequência de caracteres a partir
# do texto e o parâmetro redun_step dirá quantos
# caracteres pular entre uma sequência e outra


redun_step = 2

maxlen = 75 
drop_prob = 0.8 
lr = 1e-3
batch_size = 100

hl1 = 512 
hl2 = 256 
hl3 = 256 


##############################################################
############### Pré-processamento dos dados ##################
############################################################## 

# Vamos gerar as sequências de caracteres com a função abaixo
# Os dados gerados serão da forma:
# (n# de dados, tamanho da sequência, quantidade de
# caracteres no vocabulário)
# Para manter essa última dimensão pequena, vamos retirar
# todos os acentos e cedilhas.

X, Y, char_idx = \
	textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=redun_step)

print 'Formato dos dados: ', X.shape, Y.shape


##############################################################
############ Modelo de Rede Neural Recorrente ################
############################################################## 

# Faremos uma RNR com 3 camadas de células LSTM e 
# Dropout entre elas. Passaremos o output da última camada
# recorrente por um modelo linear e estimaremos a distribuição
# multinomial do próximo caractere, condicionada nos
# caracteres passados. Para isso, vamos minimizar a entropia
# cruzada ente a probabilidade do próximo caractere
# (calculada com uma função softmax) e o caractere de fato 
# observado. Vamos utilizar o otimizados Adam.

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, hl1, return_seq=True)
g = tflearn.dropout(g, drop_prob)
g = tflearn.lstm(g, hl2, return_seq=True)
g = tflearn.dropout(g, drop_prob)
g = tflearn.lstm(g, hl3)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
					   learning_rate=lr)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
							  seq_maxlen=maxlen,
							  clip_gradients=5.0)




##############################################################
################### Loop de treinamento ######################
############################################################## 

# retirar o comentário para carregar rede previamente treinada
#m.load(save_file)


# Define o número de vezes que a rede neural observara todas
# as Sequências
for _ in range(epochs):
	
	# Treina em mini-lotes e utiliza todos os dados
	try:
		m.fit(X, Y, validation_set=0.01, batch_size=batch_size,
			  n_epoch=1, snapshot_epoch=True, run_id='LiterNet')
		
	except KeyboardInterrupt:
		# aborta com ctrl+c
		break

	m.save(save_file) # salva o modelo

	# Cria uma sequência de caracteres a partir do texto
	# A RNR utilizará essa sequência como ponto de partida
	seed = random_sequence_from_textfile(path, maxlen) 
	
	# Gera texto. Começaremos apresentando a sequência
	# feita acima. A seguir, a rede nós dará o próximo
	# caractere. Nós então colocaremos esse caractere no
	# final da sequência de inicialização e continuaremos
	# prevendo mais e mais caracteres.

	print "\n\n\n-- Testando..."
	print "\n-- Teste com temperatura de 1.0 --\n"
	print m.generate(1000, temperature=1.0, seq_seed=seed)
	
	print "\n-- Teste com temperatura de 0.5 --\n"
	print m.generate(1000, temperature=0.5, seq_seed=seed)

	print "\n-- Teste com temperatura de 0.25 --\n"
	print m.generate(1000, temperature=0.25, seq_seed=seed)