import torch
import torch.onnx

# Define a simple test model
class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=5, embedding_dim=128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.seq(x)


torch.manual_seed(123)
model = TestModel()
model.eval()

# test data
X_test = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64).transpose(0, 1)
y = model(X_test)
print(y)

torch.onnx.export(model, X_test, "test_model.onnx", input_names=['input'], output_names=['output'])

# torch.save(
#     model.state_dict(),
#     "models/onnx_test.pt"
# )