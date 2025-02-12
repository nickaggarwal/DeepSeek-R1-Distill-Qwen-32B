from threading import Thread
from inferless import Cls # Add the inferless library


InferlessCls = Cls(gpu="T4")  # Init the class. the type of GPU you want to run with (This can be passed from the command argument)
class InferlessPythonModel:

    @InferlessCls.load     # Add the annotation
    def initialize(self):
        import torch
        from transformers import pipeline
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)

    @InferlessCls.infer    # Add the annotation
    def infer(self, inputs):
        prompt = inputs['message']
        pipeline_output = self.generator(prompt, do_sample=True, min_length=120)
        generated_txt = pipeline_output[0]["generated_text"]
        return {"generated_text": generated_txt }
