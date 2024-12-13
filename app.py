import os
import time
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from pathlib import Path
import json
from dotenv import load_dotenv
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.text_content_block import TextContentBlock
from openai.types.beta.threads.image_file_content_block import ImageFileContentBlock

load_dotenv(dotenv_path='/workspaces/azureai-assistant-tool/local.env', override=True)

# Create the AOAI client to use for the proxy agent.
from openai import AzureOpenAI

assistant_client = AzureOpenAI(
    api_key=os.getenv("GPT4_AZURE_OPENAI_KEY"),  # Your API key for the assistant api model
    api_version=os.getenv("GPT4_AZURE_OPENAI_API_VERSION"),  # API version (i.e. 2024-02-15-preview)
    azure_endpoint=os.getenv("GPT4_AZURE_OPENAI_ENDPOINT"),  # Your Azure endpoint (i.e. "https://YOURENDPOINT.openai.azure.com/")
)
# Assistant model should be '1106' or higher
assistant_deployment_name = os.getenv("GPT4_DEPLOYMENT_NAME")  # The name of your assistant model deployment in Azure OpenAI (i.e. "GPT4Assistant")

# name of the model deployment for DALL·E 3
dalle_client = AzureOpenAI(
    api_key=os.getenv("DALLE3_AZURE_OPENAI_KEY"),
    api_version=os.getenv("DALLE3_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("DALLE3_AZURE_OPENAI_ENDPOINT"),
)
dalle_deployment_name = os.getenv("DALLE3_DEPLOYMENT_NAME")

# name of the model deployment for GPT 4 with Vision
vision_client = AzureOpenAI(
    api_key=os.getenv("GPT4VISION_AZURE_OPENAI_KEY"),
    api_version=os.getenv("GPT4VISION_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("GPT4VISION_AZURE_OPENAI_ENDPOINT"),
)
vision_deployment_name = os.getenv("GPT4VISION_DEPLOYMENT_NAME")

# Create an assistant for image generation
name_dl = "dalle_assistant"
instructions_dl = """As a premier AI specializing in image generation, you possess the expertise to craft precise visuals based on given prompts. It is essential that you diligently generate the requested image, ensuring its accuracy and alignment with the user's specifications, prior to delivering a response."""
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Creates and displays an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt to be used to create the image",
                    }
                },
                "required": ["prompt"],
            },
        },
    }
]

verbose_output = True

dalle_assistant = assistant_client.beta.assistants.create(
    name=name_dl,
    instructions=instructions_dl,
    model=assistant_deployment_name,
    tools=tools
)

def generate_image(prompt: str) -> str:
    """
    Call the Azure OpenAI Dall-e 3 model to generate an image from a text prompt.
    Executes the call to the Azure OpenAI Dall-e 3 image creator, saves the file into the local directory, and displays the image.
    """
    print("Dalle Assistant Message: Creating the image ...")

    response = dalle_client.images.generate(
        model=dalle_deployment_name, prompt=prompt, size="1024x1024", quality="standard", n=1
    )

    # Retrieve the image URL from the response (assuming response structure)
    image_url = response.data[0].url

    # Open the image from the URL and save it to a temporary file.
    im = Image.open(requests.get(image_url, stream=True).raw)

    # Define the filename and path where the image should be saved.
    filename = "temp.jpg"
    local_path = Path(filename)

    # Save the image.
    im.save(local_path)

    # Get the absolute path of the saved image.
    full_path = str(local_path.absolute())

    img = cv2.imread("temp.jpg", cv2.IMREAD_UNCHANGED)

    # Convert the image from BGR to RGB for displaying with matplotlib,
    # because OpenCV uses BGR by default and matplotlib expects RGB.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with matplotlib.
    plt.imshow(img_rgb)
    plt.axis("off")  # Turn off axis labels.
    plt.show()

    # Return the full path of the saved image.
    print("Dalle Assistant Message: " + full_path)
    return "Image generated successfully and store in the local file system. You can now use this image to analyze it with the vision_assistant"

# Create an assistant for image analysis
name_vs = "vision_assistant"
instructions_vs = """As a leading AI expert in image analysis, you excel at scrutinizing and offering critiques to refine and improve images. Your task is to thoroughly analyze an image, ensuring that all essential assessments are completed with precision before you provide feedback to the user. You have access to the local file system where the image is stored."""
tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "analyzes and critics an image",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]

vision_assistant = assistant_client.beta.assistants.create(
    name=name_vs, instructions=instructions_vs, model=assistant_deployment_name, tools=tools
)

def analyze_image() -> str:
    """
    Call the Azure OpenAI GPT4 Vision model to analyze and critic an image and return the result. The resulting output should be a new prompt for dall-e that enhances the image based on the criticism and analysis.
    """
    print("Vision Assistant Message: " + "Analyzing the image...")

    import base64
    from pathlib import Path

    # Create a Path object for the image file
    image_path = Path("/workspaces/azureai-assistant-tool/temp.jpg")

    # Using a context manager to open the file with Path.open()
    with image_path.open("rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    content_images = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        for base64_image in [base64_image]
    ]
    response = vision_client.chat.completions.create(
        model=vision_deployment_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze and critic this image and generate a new enhanced prompt for Dall-e with the criticism and analysis.",
                    },
                    *content_images,
                ],
            }
        ],
        max_tokens=1000,
    )
    print("Vision Assistant Message: " + response.choices[0].message.content)
    return response.choices[0].message.content

# User proxy assistant to streamline dialogue
name_pa = "user_proxy"
agent_arr = ["dalle_assistant", "vision_assistant"]
agent_string = ""
for item in agent_arr:
    agent_string += f"{item}\n"

instructions_pa = f"""As a user proxy agent, your primary function is to streamline dialogue between the user and the specialized agents within this group chat. You are tasked with articulating user inquiries with clarity to the relevant agents and maintaining a steady flow of communication to guarantee the user's request is comprehensively addressed. Please withhold your response to the user until the task is completed, unless an issue is flagged by the respective agent or when you can provide a conclusive reply.

You have access to the local file system where files are stored. For example, you can access the image generated by the Dall-e assistant and send it to the Vision assistant for analysis.

You have access to the following agents to accomplish the task:
{agent_string}
If the agents above are not enough or are out of scope to complete the task, then run send_message with the name of the agent.

When outputting the agent names, use them as the basis of the agent_name in the send message function, even if the agent doesn't exist yet.

Run the send_message function for each agent name generated. 

Do not ask for followup questions, run the send_message function according to your initial input.
"""

tools = [
    {"type": "code_interpreter"},
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send messages to other agents in this group chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The message to be sent",
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "The name of the agent to execute the task.",
                    },
                },
                "required": ["query", "agent_name"],
            },
        },
    },
]

verbose_output = True
user_proxy = assistant_client.beta.assistants.create(
    name=name_pa, instructions=instructions_pa, model=assistant
