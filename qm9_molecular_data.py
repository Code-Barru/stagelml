
from math import sqrt
from sklearn.pipeline import FeatureUnion
import torch
from torch.nn import Linear, Parameter, Sequential, Module
from torch.utils.tensorboard import SummaryWriter
import torch_cluster
import torch_geometric

from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.glob import global_add_pool
from torch_geometric.utils import add_self_loops

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")


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

	def forward(self, x, h, edge_index):		
		norm2=(x[edge_index[0]]-x[edge_index[1]]).norm(dim=1).pow(2)
		return self.propagate(edge_index, x=h, norm2=norm2)

	def message(self,x_i:torch.Tensor,x_j:torch.Tensor, norm2:torch.Tensor):
		inputs = torch.cat((x_i,x_j,norm2.unsqueeze(1)),dim=1)
		return self.f_message(inputs)
			
	def update(self, m_i:torch.Tensor, x:torch.Tensor):
		inputs = torch.cat((m_i,x),dim=1)

		return self.f_update(inputs)+x


class EGNN(torch.nn.Module):
	def __init__(self,features=128, feature_message=16, n_layer=7, hidden_features=128, target_dim=12) -> None:
		super().__init__()
		
		self.embeding = torch.nn.Embedding(5,features)
		self.gnn = torch.nn.ModuleList([
			GCNConv(features=features, features_message=feature_message) for _ in range(n_layer)
		])

		self.node_proj = torch.nn.Sequential(
			Linear(features, hidden_features),
			Swish(),
			Linear(hidden_features, hidden_features)
		)
		
		self.readout = torch.nn.Sequential(
			Linear(features, hidden_features),
			Swish(),
			Linear(hidden_features, target_dim)
		)

		self.atomic_number = torch.nn.Parameter(torch.tensor([-1,0,-1,-1,-1,-1,1,2,3,4],dtype=torch.long),requires_grad=False)

	def forward(self, batch):
		
		x, z, edge_index, batch = batch.pos, batch.z, batch.edge_index, batch.batch

		h = self.embeding(self.atomic_number[z])
		for layer in self.gnn:
			h = layer(x,h,edge_index)

		projection = self.node_proj(h)
		sum_pooling = global_add_pool(projection, batch=batch)
		return self.readout(sum_pooling)


def train_once(dataloader, model, loss_f, optimizer, mean, std):
		size =  int(len(dataloader.dataset)/64)
		model.train()
		loss = 0

		for (i,batch) in enumerate(dataloader):
			batch = batch.to(device)

			pred = model(batch)

			idx_index = torch.tensor([5,8,6,7,4,15,14,13,9,12,11,10])
			# "alpha", "gap", "homo", "lumo", "mu", "cv", "g298", "h298", "u298", "u0", "zpve"
			loss = loss_f(pred, (batch.y[:,idx_index]-mean)/std )
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 100 == 0:
				loss = loss.item()
				print(f"batch [{i:>5d}/{size:>5d}]")


def test(dataloader, model, loss_f, mean, std):
	size = int(len(dataloader.dataset)/64)
	num_batches = len(dataloader)
	model.eval()
	test_loss, accuracy = 0, 0

	print(f"Testing model.")
	idx_index = torch.tensor([5,8,6,7,4,15,14,13,9,12,11,10]).to(device)
	# "alpha", "gap", "homo", "lumo", "mu", "cv", "g298", "h298", "u298", "u0", "zpve"

	with torch.no_grad():
		for (i,batch) in enumerate(dataloader):

			batch = batch.to(device)

			pred = model(batch)
			
			batch.y = batch.y[:,idx_index]

			test_loss += loss_f(pred, (batch.y - mean) / std).item()
			
			pred = (pred*std) + mean
			accuracy += (pred[:,10] - batch.y[:,10]).abs().sum() / len(dataloader.dataset)

			if i%100 == 0:
				print(f"[{i:>5d}/{size:>5d}]")

	test_loss /= num_batches

	return accuracy, test_loss


def train_epochs(epochs, model, save_name, batch_size, 
	optimizer, loss_fn=torch.nn.L1Loss()):

	training_data = datasets.QM9(
		root="data",
	)

	size = len(training_data)
	tmp = torch.utils.data.random_split(training_data, [int(0.8 * size), size - int(0.8 * size)], generator=torch.Generator().manual_seed(10))

	training_loader = DataLoader(tmp[0],batch_size)
	testing_loader = DataLoader(tmp[1],batch_size)

	std, mean = calcul_standard_deviation(training_loader)

	std = std.to(device)
	mean = std.to(device)

	writer = SummaryWriter()
	for epoch in range(epochs):
		print(f'Epoch [{epoch+1:>3d}/{epochs:>3d}]')
		
		train_once(training_loader, model, loss_fn, optimizer, std, mean)
		accuracy, loss = test(testing_loader, model, loss_fn, std, mean)

		writer.add_scalar("Accuracy/epoch", accuracy, epoch+1)
		writer.add_scalar("Loss/epoch", loss, epoch+1)

	writer.flush()
	writer.close()
	print("Done!")
	torch.save(model.state_dict(), save_name)
	print(f"Saved PyTorch Model state to {save_name}.")

def calcul_standard_deviation(dataloader):
	idx_index = torch.tensor([5,8,6,7,4,15,14,13,9,12,11,10])
	# "alpha", "gap", "homo", "lumo", "mu", "cv", "g298", "h298", "u298", "u0", "zpve"
	tmp = []
	for (i,batch) in enumerate(dataloader):
		batch.y = batch.y[:,idx_index]
		tmp.append(batch.y)

	tmp = torch.cat(tmp,0)
	
	return torch.std(tmp,0), torch.mean(tmp,0)


if __name__=="__main__":
	model = EGNN()
	model.to(device)

	train_epochs(25, model, "model.pth", 64, torch.optim.Adam(model.parameters(), lr=1e-5))