class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.reshape(batch_size, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(in_features=784, hidden_features=256, num_classes=10).to("cuda")

with torch.no_grad():
    fan_in_fc1 = model.fc1.weight.size(1)
    scale_fc1 = (6.0 / fan_in_fc1) ** 0.5
    model.fc1.weight.uniform_(-scale_fc1, scale_fc1)
    model.fc1.bias.zero_()

    fan_in_fc2 = model.fc2.weight.size(1)
    scale_fc2 = (6.0 / fan_in_fc2) ** 0.5
    model.fc2.weight.uniform_(-scale_fc2, scale_fc2)
    model.fc2.bias.zero_()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)