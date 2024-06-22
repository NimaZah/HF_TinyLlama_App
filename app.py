import gradio as gr
from transformers import pipeline

# Replace with your model's path
model_name = "NimaZahedinameghi/nimaAxolotl"

# Initialize the chat pipeline
chat_pipeline = pipeline("text-generation", model=model_name)

def chat_with_model(user_input):
    # Generate a response from the model
    response = chat_pipeline(user_input, max_length=100, do_sample=True, top_p=0.95, top_k=50)
    return response[0]['generated_text']

# Create the Gradio interface
interface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.inputs.Textbox(lines=7, placeholder="Enter your message here..."),
    outputs="text",
    title="Chat with Fine-tuned Model",
    description="Interact with the fine-tuned chat model."
)

if __name__ == "__main__":
    interface.launch()
