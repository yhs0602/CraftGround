import torch
from torch.autograd import Variable
from torchsummary import summary
from torchviz import make_dot

from models.dqn import CNNDQN, SoundDQN


def visualize_cnn():
    hidden_dim = 128  # 128정도 해보기
    kernel_size = 5
    stride = 2
    state_dim = (3, 114, 64)
    action_dim = 6
    # Create an instance of your model
    model = CNNDQN(state_dim, action_dim, kernel_size, stride, hidden_dim)

    # Generate a dummy input tensor
    x = Variable(torch.rand(1, *state_dim))

    # Visualize the model
    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render("model_graph", format="png")  # Save the graph as an image
    summary(model, state_dim)


def visualize_fcn():
    hidden_dim = 128  # 128정도 해보기
    state_dim = (7,)
    action_dim = 6
    model = SoundDQN(state_dim, action_dim, hidden_dim)
    model.eval()
    x = Variable(torch.rand(1, *state_dim))
    dot = make_dot(model(x), params=dict(model.named_parameters()))
    dot.render("sound_dqn_model_graph", format="png")  # Save the graph as an image
    summary(model, state_dim)


if __name__ == "__main__":
    visualize_cnn()
    visualize_fcn()
