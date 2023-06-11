from torchview import draw_graph

from models.dqn import CNNDQN, SoundDQN, DQNSoundAndVision


def visualize_cnn():
    hidden_dim = 128  # 128정도 해보기
    kernel_size = 5
    stride = 2
    state_dim = (3, 114, 64)
    action_dim = 6
    # Create an instance of your model
    model = CNNDQN(state_dim, action_dim, kernel_size, stride, hidden_dim)

    model_graph = draw_graph(model, input_size=(256, *state_dim), device="meta")
    print(model_graph.visual_graph)
    print(dir(model_graph.visual_graph))
    with open("cnn.dot", "w") as f:
        f.write(model_graph.visual_graph.source)


def visualize_fcn():
    hidden_dim = 128  # 128정도 해보기
    audio_dim = (7,)
    action_dim = 6
    # Create an instance of your model
    model = SoundDQN(audio_dim, action_dim, hidden_dim)

    model_graph = draw_graph(model, input_size=(256, *audio_dim), device="meta")
    print(model_graph.visual_graph)
    print(dir(model_graph.visual_graph))
    with open("fcn.dot", "w") as f:
        f.write(model_graph.visual_graph.source)


def visualize_both():
    hidden_dim = 128
    kernel_size = 5
    stride = 2
    video_dim = (3, 114, 64)
    audio_dim = (7,)
    action_dim = 6
    # Create an instance of your model
    model = DQNSoundAndVision(
        video_dim, audio_dim, action_dim, kernel_size, stride, hidden_dim
    )

    model_graph = draw_graph(
        model, input_size=[(256, *video_dim), (256, *audio_dim)], device="meta"
    )
    print(model_graph.visual_graph)
    print(dir(model_graph.visual_graph))
    with open("both.dot", "w") as f:
        f.write(model_graph.visual_graph.source)


if __name__ == "__main__":
    visualize_cnn()
    visualize_fcn()
    visualize_both()
