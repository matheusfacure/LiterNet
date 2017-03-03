# -*- coding: latin-1 -*-
from __future__ import absolute_import, division

import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

path = "copilados"
save_file = 'cpk_literNET/LiterNet'

# hiper-parâmetros
maxlen = 75
drop_prob = 0.8
lr = 1e-3
batch_size = 100
redun_step = 2

hl1 = 512 
hl2 = 256 
hl3 = 256 


X, Y, char_idx = \
	textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=redun_step)

print 'Formato dos dados: ', X.shape, Y.shape

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


#m.load(save_file)

for i in range(100):
	
	seed = random_sequence_from_textfile(path, maxlen)
	
	try:
		m.fit(X, Y, validation_set=0.01, batch_size=batch_size,
			  n_epoch=1, snapshot_epoch=True, run_id='LiterNet')
		
	except KeyboardInterrupt:
		break

	m.save(save_file)
	print "\n\n\n-- Testando..."
	
	print "\n-- Teste com temperatura de 1.0 --\n"
	print m.generate(1000, temperature=1.0, seq_seed=seed)
	
	print "\n-- Teste com temperatura de 0.5 --\n"
	print m.generate(1000, temperature=0.5, seq_seed=seed)

	print "\n-- Teste com temperatura de 0.25 --\n"
	print m.generate(1000, temperature=0.25, seq_seed=seed)