import torch
from torch import nn
from torch.sparse import FloatTensor
from torch.optim import Adam
torch.manual_seed(2023)

class Temp(nn.Module):
    def __init__(self) -> None:
        super(Temp, self).__init__()
        self.weights = nn.Parameter(torch.randn(3))
        self.sp_mat = FloatTensor(torch.tensor([[0, 1, 2], [2, 1, 3]]), self.weights, (3, 4))
        # self.sp_mat = nn.Parameter(torch.randn(3, 4))

    def forward(self, x):
        # sp_mat = FloatTensor(torch.tensor([[0, 1, 2], [2, 1, 3]]), self.weights, (3,4))
        return x @ self.sp_mat
    
    def cal_loss(self, x):
        output = self.forward(x)
        return torch.mean(output @ output.T)
    
    


t = Temp()
print(list(t.parameters()))
opt = Adam(t.parameters(), lr=0.01)
x1 = torch.tensor([[1., 1., 2.]], requires_grad=False)
x2 = torch.tensor([[1., 1., 1.]], requires_grad=False)

for e in range(100):
    t.train(True)
    for i in [x1, x2]:
        opt.zero_grad()
        loss = t.cal_loss(i)
        loss.backward(retain_graph=True)
        opt.step()

        loss.detach()

t.eval()
print(list(t.parameters()))


'''
chỉ với sparse matrix
kết quả khi dùng retain_graph = true/false là giống nhau

với false: phải dựng lại sp matrix ở forward
với true: dụng sp matrix ở trong init cũng được
'''

