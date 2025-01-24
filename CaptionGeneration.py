# Function to create a prompt
import pandas as pd
from PIL import Image

from RobustSam import RobustSam
from EVA02 import Eva
from LLMHuggingFaceClass import LLAMA
from Owl import Owl
from RamPlusPlus import RamPlus


def define_models():
    models = [
        LLAMA("LLAMA32"),
        Owl("Owl"),
        #Eva("EVA"),
        RobustSam("robustSAM"),
        RamPlus("ram_plus_swin_large_14m")
    ]
    return models

def create_prompt_info(image: Image ,model_list: list):
    result={}
    for model in model_list:
        result[model.config["GenerationType"]]+=model.generateResponse(image)
    return result

def create_prompt(data):
    prompt = f"""
    You are a caption generation model. Based on the details provided below, 
    generate a concise and accurate caption for the image. Only describe what is mentioned in the details.
     Avoid any additional or imagined information and keep the caption short and relevant.

    Details:
    - Detected objects: {', '.join(data['Detected objects'])}
    - Scene context: {', '.join(data['Scene context'])}
    - OCR text: {data['OCR text'] if data['OCR text'] else 'None'}

    Caption:
    """
    return prompt


# Example inputs for testing
example_inputs = [
    {
        "Detected objects": ["dog", "ball"],
        "Scene context": ["outdoor", "playing"],
        "OCR text": ""
    },
    {
        "Detected objects": ["car", "person"],
        "Scene context": ["urban", "traffic"],
        "OCR text": "Speed Limit 60"
    },
    {
        "Detected objects": ["cake", "table"],
        "Scene context": ["indoor", "celebration"],
        "OCR text": "Happy Birthday"
    }
]


# Function to process inputs and generate captions
def test_pipeline(inputs, llm):
    results = []

    for idx, data in enumerate(inputs):
        # Create prompt
        prompt = create_prompt(data)
        print(f"Prompt for Example {idx + 1}:\n{prompt}\n{'-' * 50}")  # Debug prompt
        # Generate caption
        output = llm.generateResponse(
            prompt
        )

        # Extract the actual caption
        caption_start = output.find("Caption:") + len("Caption:")
        caption = output[caption_start:].strip() if caption_start > 0 else output.strip()

        # Save results
        results.append({"Input": data, "Caption": caption})

        # Print results for quick inspection
        print(f"Example {idx + 1}:\nInput: {data}\nGenerated Caption: {caption}\n{'-' * 50}")
    return results

def secondTest(llm):
    models = define_models()
    image = Image.open("image/AMBER_2.jpg")
    data=create_prompt_info(image,models)
    prompt = create_prompt(data)
    print(f"Prompt for Example:\n{prompt}\n{'-' * 50}")  # Debug prompt
    # Generate caption
    output = llm.generateResponse(
        prompt
    )

    # Extract the actual caption
    caption_start = output.find("Caption:") + len("Caption:")
    caption = output[caption_start:].strip() if caption_start > 0 else output.strip()

    # Save results
    result={"Input": data, "Caption": caption}

    # Print results for quick inspection
    print(f"Example:\nInput: {data}\nGenerated Caption: {caption}\n{'-' * 50}")
    return result

if __name__=="__main__":
    llmmodel=LLAMA("LLAMA32")
    # Run the test
    results = secondTest(llmmodel)

    # Save results to a CSV for analysis
    df = pd.DataFrame(results)
    df.to_csv("generated_captions.csv", index=False)
    print("Results saved to 'generated_captions.csv'")