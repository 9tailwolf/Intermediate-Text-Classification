import torch
from tqdm import tqdm
import time

class Training:
    def __init__(self,model,data):
        self.device = torch.device(self.set_device())
        self.model = model.to(self.device)
        self.data = data.to(self.device)

    def training(self,lr,epochs,weight_decay=5e-4):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            time.sleep(1e-5)
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = torch.nn.functional.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()


        self.eval(loss.item())

    def set_device(self):
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def eval(self,loss):
        self.model.eval()
        pred = self.model(self.data).argmax(dim=1)
        correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()
        acc = int(correct) / int(self.data.test_mask.sum())
        print('Loss : ', loss, ', Accuracy : ', acc)




