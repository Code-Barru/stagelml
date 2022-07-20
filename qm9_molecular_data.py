from nbformat import write
import torch
from torch.nn import Linear, Parameter, Sequential, Module
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.glob import global_add_pool
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

	def forward(self, x, h, edge_index):		
		norm2=(x[edge_index[0]]-x[edge_index[1]]).norm(dim=1).pow(2)
		return self.propagate(edge_index, x=h, norm2=norm2)

	def message(self,x_i:torch.Tensor,x_j:torch.Tensor, norm2:torch.Tensor):
		inputs = torch.cat((x_i,x_j,norm2.unsqueeze(1)),dim=1)
		return self.f_message(inputs)
			
	def update(self, m_i:torch.Tensor, x:torch.Tensor):
		inputs = torch.cat((m_i,x),dim=1);

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


def train_once(dataloader, model, loss_f, optimizer, device="cpu"):
		size = len(dataloader.dataset)
		model.train()
		loss = 0

		for (i,batch) in enumerate(dataloader):
			batch = batch.to(device)

			pred = model(batch)

			print(batch)

			print(batch.y.shape)
			print(pred.shape)

			loss = loss_f(pred,batch.y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch % 100 == 0:
				loss = loss.item()
				print(f"batch [{i:>5d}/{size:>5d}]")


def test(dataloader, model, loss_f):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()
	test_loss, correct = 0, 0

	actual = []
	predicted = []

	with torch.no_grad():
		for (i,batch) in enumerate(dataloader):

			pred = model(batch)

			actual.append(batch.y)
			predicted.append(pred)

			test_loss += loss_f(pred, batch.y).item()
			correct += (pred == batch.y).type(torch.float).sum().item()

	actual = torch.cat(actual).cpu()
	predicted = torch.cat(predicted).cpu()

	test_loss /= num_batches
	correct /= size

	return 100*correct, test_loss


def train_epochs(epochs, model, save_name, batch_size, 
	optimizer, loss_fn=torch.nn.CrossEntropyLoss()):

	training_data = datasets.QM9(
	root="data",
	)

	loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device.")

	writer = SummaryWriter()
	for epoch in range(epochs):
		print(f'Epoch [{epoch:>3d}/{epoch:>3d}]')
		accuracy, loss = train_once(loader, model, loss_fn, optimizer)

		writer.add_scalar("Accuracy/epoch", accuracy, epoch+1)
		writer.add_scalar("Loss/epoch", loss, epoch+1)

	writer.flush()
	writer.close()
	print("Done!")
	torch.save(model.state_dict(), save_name)
	print(f"Saved PyTorch Model state to {save_name}.")

model = EGNN()

train_epochs(25, model, "model.pth", 64, torch.optim.SGD(model.parameters(), lr=1e-3))
