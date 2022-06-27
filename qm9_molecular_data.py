from torch_geometric.loader import DataLoader
from torch_geometric import datasets

import torch
from torch.nn import Linear, Parameter, Sequential, Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class Swish(Module):
	def __init__(self):
		super().__init__()

	def forward(self, x:torch.Tensor):
		return x*x.sigmoid()

class GCNConv(MessagePassing):
	def __init__(self, features, features_message):
		super().__init__(aggr='add')

		#self.reset_parameters()

		message_input_dim = 2*features + 1
		self.f_message = Sequential(
			Linear(message_input_dim, message_input_dim*2),
			Swish(),
			Linear(message_input_dim*2,features_message),
			Swish()
		)

		self.f_update = Sequential(
			Linear(features+features_message,2*features),
			Swish(),
			Linear(2*features,features)
		)

	def reset_parameters(self):
		self.lin.reset_parameters()
		self.bias.data.zero_()

	def forward(self, x, edge_index):		
		norm2=(x[edge_index[0]]-x[edge_index[1]]).norm(dim=1).pow(2)
		return self.propagate(edge_index, x=x,norm2=norm2)

	def message(self,x_i:torch.Tensor,x_j:torch.Tensor, norm2:torch.Tensor):
		inputs = torch.cat((x_i,x_j,norm2.unsqueeze(1)),dim=1)
		return self.f_message(inputs)
			
	def update(self, inputs: torch.Tensor):

		print(inputs.shape)

		output = self.f_update(inputs)
		
		print(output)

		return 0




training_data = datasets.QM9(
	root="data",
)

loader = DataLoader(training_data, batch_size=1, shuffle=True)

test = GCNConv(128,16)

n=1024
n_edges=2048
edges_index = torch.randint(0,n,(2,n_edges),dtype=torch.long)
x = torch.randn(n,128)

test(x,edges_index)
