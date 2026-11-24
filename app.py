import gradio as gr
import spaces
import torch

# This tensor is allocated outside the GPU-decorated function.
# On CPU-only hardware, it remains on CPU when the app starts.
zero = torch.tensor([0])


@spaces.GPU
def greet(n):
    # Move the tensor onto the GPU when Spaces executes this function.
    gpu_zero = zero.cuda()
    return f"Hello {gpu_zero + n} Tensor"


demo = gr.Interface(
    fn=greet,
    inputs=gr.Number(),
    outputs=gr.Text(),
)

demo.launch()
