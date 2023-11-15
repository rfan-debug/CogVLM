from typing import Callable

import gradio as gr
import os, sys
import torch.distributed
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import json
import torch
import time
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size

from utils.parser import parse_response
from utils.chat import chat
from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor

DESCRIPTION = '''<h2 style='text-align: center'> <a href="https://github.com/THUDM/CogVLM">CogVLM-17B</a> </h2>'''

NOTES = 'This app is adapted from <a href="https://github.com/THUDM/CogVLM">https://github.com/THUDM/CogVLM</a>. It would be recommended to check out the repo if you want to see the detail of our model.'

MAINTENANCE_NOTICE1 = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.<br>Hint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'

GROUNDING_NOTICE = 'Hint: When you check "Grounding", please use the <a href="https://github.com/THUDM/CogVLM/blob/main/utils/template.py#L344">corresponding prompt</a> or the examples below.'

default_chatbox = [("", "Hi, What do you want to know about this image?")]


def process_image_without_resize(image_prompt):
    image = Image.open(image_prompt)
    timestamp = int(time.time())
    file_ext = os.path.splitext(image_prompt)[1]
    filename_grounding = f"examples/{timestamp}_grounding{file_ext}"
    return image, filename_grounding


def load_model(args, rank, world_size):
    print(f"Loading model from {rank} with world size {world_size}\n")
    model, model_args = CogVLMModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            model_parallel_size=world_size,
            mode='inference',
            skip_init=True,
            use_gpu_initialization=True if torch.cuda.is_available() else False,
            device=f'cuda',
            **vars(args)
        ),
        overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
    )
    model = model.eval()
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    print(f"Model loading finished from {rank} with world size {world_size}\n.")
    return model, image_processor, text_processor_infer


def clear_fn(value):
    return "", default_chatbox, None


def clear_fn2(value):
    return default_chatbox


def test_run(call_predict: Callable):
    print(f"This rank: {rank}, running the gradio UI")

    input_text, result_text, hidden_image_hash = call_predict(
        input_text="Describe this image",
        temperature=0.7,
        top_p=0.4,
        top_k=10,
        image_prompt="./examples/6.jpg",
        result_previous=[['', 'Hi, What do you want to know about this image?']],
        hidden_image=None,
    )

    print("input_text: ", input_text)
    print("result_text: ", result_text)
    print("hidden_image_hash: ", hidden_image_hash)

def main(args,
         model,
         image_processor,
         text_processor_infer,
         rank,
         ):
    is_grounding = 'grounding' in args.from_pretrained
    gr.close_all()

    def call_predict(
            input_text,
            temperature,
            top_p,
            top_k,
            image_prompt,
            result_previous,
            hidden_image,
    ):
        parameters_str = f"""
            input params:
            input_text: {input_text}
            temperature: {temperature}
            top_p: {top_p}
            top_k: {top_k}
            image_prompt: {image_prompt}
            result_previous: {result_previous}
            hidden_image: {hidden_image}
            rank: {rank}
        """

        print(
            parameters_str
        )
        result_text = [(ele[0], ele[1]) for ele in result_previous]
        for i in range(len(result_text) - 1, -1, -1):
            if result_text[i][0] == "" or result_text[i][0] == None:
                del result_text[i]
        print(f"history {result_text}")

        try:
            with torch.no_grad():
                pil_img, image_path_grounding = process_image_without_resize(image_prompt)

                # Pull all necessary data into list so we can broadcast them through `torch.distributed.broadcast_object_list`.
                if rank == 0:
                    image_prompts = [image_prompt]
                    input_texts = [input_text]
                    pil_imgs = [pil_img]
                else:
                    image_prompts = [None]
                    input_texts = [None]
                    pil_imgs = [None]


                print("result_text: ", result_text)

                if world_size > 1:
                    torch.distributed.broadcast_object_list(input_texts, src=0)
                    if len(result_text) > 0:
                        torch.distributed.broadcast_object_list(result_text, src=0)
                    torch.distributed.broadcast_object_list(image_prompts, src=0)
                    torch.distributed.broadcast_object_list(pil_imgs, src=0)

                    print("image_prompts:", image_prompts)
                    print("result_texts:", result_text)
                    print("input_texts:", input_texts)
                    pil_img.save(f"temp_{rank}.png")

                print(f"Calling chat from rank={rank}")
                response, _, cache_image = chat(
                    image_path=image_prompts[0],
                    model=model,
                    text_processor=text_processor_infer,
                    img_processor=image_processor,
                    query=input_texts[0],
                    history=result_text,
                    image=None,
                    max_length=2048,
                    top_p=top_p,
                    temperature=temperature,
                    top_k=top_k,
                    invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer, "invalid_slices") else [],
                    no_prompt=False
                )
                print("chat call finished")
        except Exception as e:
            print("error message", e)
            traceback.print_exc()
            result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
            return "", result_text, hidden_image

        answer = response
        if is_grounding:
            parse_response(pil_img, answer, image_path_grounding)
            new_answer = answer.replace(input_text, "")
            result_text.append((input_text, new_answer))
            result_text.append((None, (image_path_grounding,)))
        else:
            result_text.append((input_text, answer))

        print(result_text)
        print('Chat task finished')
        return "", result_text, hidden_image


    if rank == 0:
        with gr.Blocks(css='style.css') as demo:
            # Design the interface:
            gr.Markdown(DESCRIPTION)
            gr.Markdown(NOTES)

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Group():
                        input_text = gr.Textbox(label='Input Text',
                                                placeholder='Please enter text prompt below and press ENTER.')
                        with gr.Row():
                            run_button = gr.Button('Generate')
                            clear_button = gr.Button('Clear')

                        image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)

                    with gr.Row():
                        temperature = gr.Slider(maximum=1, value=0.8, minimum=0, label='Temperature')
                        top_p = gr.Slider(maximum=1, value=0.4, minimum=0, label='Top P')
                        top_k = gr.Slider(maximum=100, value=10, minimum=1, step=1, label='Top K')

                with gr.Column(scale=5):
                    result_text = gr.components.Chatbot(
                        label='Multi-round conversation History',
                        value=[("", "Hi, What do you want to know about this image?")]).style(height=550)
                    hidden_image_hash = gr.Textbox(visible=False)

            gr.Markdown(MAINTENANCE_NOTICE1)

            print("image_prompt", image_prompt)

            # Add the trigger
            run_button.click(fn=call_predict,
                             inputs=[input_text, temperature, top_p, top_k, image_prompt, result_text, hidden_image_hash],
                             outputs=[input_text, result_text, hidden_image_hash])
            input_text.submit(fn=call_predict,
                              inputs=[input_text, temperature, top_p, top_k, image_prompt, result_text, hidden_image_hash],
                              outputs=[input_text, result_text, hidden_image_hash])
            clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, image_prompt])
            image_prompt.upload(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
            image_prompt.clear(fn=clear_fn2, inputs=clear_button, outputs=[result_text])
            print(f"Gradio version: {gr.__version__}")

        demo.queue(concurrency_count=10)
        demo.launch()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', None))
    print(f"local_rank: {local_rank}")
    parser = CogVLMModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # Load models
    model, image_processor, text_processor_infer = None, None, None # load_model(args, rank, world_size)
    main(args,
         model,
         image_processor,
         text_processor_infer,
         rank)
