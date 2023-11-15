import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.part1 = nn.Linear(10, 10).to('cuda:0')  # Part of the model on GPU 0
        self.part2 = nn.Linear(10, 10).to('cuda:1')  # Part of the model on GPU 1

    def forward(self, x):
        x = self.part1(x.to('cuda:0'))
        x = self.part2(x.to('cuda:1'))
        return x


import gradio as gr

def predict(input):
    model = MyModel()
    # ... load model weights if necessary
    output = model(torch.tensor(input))
    return output.cpu().data.numpy()

# Launch command
# torchrun --standalone --nnodes=1 --nproc-per-node=2 gradio_example.py

iface = gr.Interface(fn=predict, inputs="number", outputs="number")
iface.launch()