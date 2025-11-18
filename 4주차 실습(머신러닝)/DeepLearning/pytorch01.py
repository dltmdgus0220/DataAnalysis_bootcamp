import torch

# torch 설치 확인
# print('PyTorch version :', torch.__version__)
# print('CUDA available :', torch.cuda.is_available())
# print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')

a = torch.tensor([10.0, 20., 30., 40.])
b = torch.tensor([[1.0],
                  [2.0],
                  [3.0]])
# print(a.shape, b.shape)
c = a + b
print(c) # broadcast

x = torch.tensor([2.0], requires_grad=True) # backward를 통해 미분을 하려면 requires_grad=True 해줘야함.
y = 3 * x**2 + 2 * x
print(x)
print(y)
y.backward()
print(x.grad)