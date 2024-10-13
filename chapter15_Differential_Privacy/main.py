import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	
	with open(args.conf, 'r') as f:
		conf = json.load(f)
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
	
	print("\n\n")
	acc_store, loss_store = [], []
	for e in range(conf["global_epochs"]):
		
		candidates = random.sample(clients, conf["k"])
		
		weight_accumulator = {}
		
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		
		for c, idx in enumerate(candidates):
			diff = c.local_train(server.global_model)
			print("#{} client done".format(idx))
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
		
		server.model_aggregate(weight_accumulator)
		
		acc, loss = server.model_eval()
		acc_store.append(acc)
		loss_store.append(loss)
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
	plt.figure(dpi=160)
	plt.figure(1)
	plt.plot([i + 1 for i in range(len(acc_store))], acc_store)
	plt.figure(2)
	plt.plot([i + 1 for i in range(len(loss_store))], loss_store)
	plt.show()
