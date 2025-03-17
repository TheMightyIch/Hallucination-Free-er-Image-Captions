# Function to create a prompt
import json

import pandas as pd
from PIL import Image
import os
import importlib
import inspect

import Models.AbstractModel

MODELS_FOLDER = "Models"


def import_models(allowed_models: list[str]):
    imported_classes = {}
    # Get all Python files in the folder (excluding __init__.py if present)
    module_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith(".py") and f != "__init__.py"]

    # Dynamically import modules and extract their classes
    for model_file in module_files:
        model_name = model_file[:-3]  # Remove ".py" extension
        model_path = f"{MODELS_FOLDER}.{model_name}"  # Convert to module import path

        try:
            module = importlib.import_module(model_path)  # Import module dynamically
        except ImportError as e:
            print(f"Error importing module {model_path}: {e}")
            continue  # Skip to next module if import fails

        for class_name in allowed_models:
            try:
                obj = getattr(module, class_name, None)  # Get class by name
                if obj and inspect.isclass(obj) and issubclass(obj,
                    Models.AbstractModel.AbstractModel) and obj is not Models.AbstractModel.AbstractModel:
                    imported_classes[class_name] = obj  # Store class reference
            except Exception as e:
                print(f"Error loading class {class_name} from {model_path}: {e}")
    return imported_classes


def define_models(allowed_models: list[str]):
    imported_classes = import_models(allowed_models)
    models=[v(k) for k,v in imported_classes.items()]
    print("sucessfully loaded the following models: ", [model.model_alias for model in models])
    return models


def create_prompt_info(image: Image, model_list: list):
    result = {}
    tagGenerationModel= [model for model in model_list if model.config["GenerationType"] == "Scene context"]
    for model in tagGenerationModel:
        print("Loading model: ", model.model_alias)
        model.loadModel()
        result.setdefault(model.config["GenerationType"], []).append(model.generateResponse(image))
    otherModels=set(model_list)-set(tagGenerationModel)
    for model in otherModels:
        print("Loading model: ", model.model_alias)
        model.loadModel()
        if model.config["tag_input_needed"] == "yes":
            result.setdefault(model.config["GenerationType"], []).append(model.generateResponse(image, ['. '.join(str(object)) for object in result["Scene context"]]))
        result.setdefault(model.config["GenerationType"], []).append(model.generateResponse(image))
    return result


def create_prompt(prompt:str, data: dict[str,str])-> str:
    return prompt.format(Objects=[', '.join(str(object)) for object in data['Detected objects']],
               Context=[', '.join(str(object)) for object in data['Scene context']],
               OCR=data['OCR text'] if 'OCR text' in data.keys() else 'None',
                WorldKnowledge = data['WorldKnowledge'] if 'WorldKnowledge' in data.keys() else 'None')

def load_run_cfg(run_setting :str):
    file_name = f'{os.getcwd()}/Models/configs/{run_setting}.json'
    with open(file_name, 'r', encoding='utf-8') as cfile:
        data = json.load(cfile)
    return data


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


def secondTest():
    cfg= load_run_cfg("RunSettings")
    models = define_models(cfg["Models"])
    llm = [model for model in models if model.config["GenerationType"] == "LLM"]
    try:
        llm=(lambda x: x)(*llm)
    except Exception as e:
        print(f"Error loading LLM, either to few or to many LLMs found: {e}")
        return

    models[:] = (model for model in models if model.config["GenerationType"] != "LLM")
    if len(models)> len(set(models)):
        assert "Error: Duplicate models detected"
        return
    image = Image.open("image/AMBER_2.jpg")
    print("Image loaded")
    data = create_prompt_info(image, models)
    print("Prompt info created")
    prompt = create_prompt(cfg["Prompt"],data)
    print(f"Prompt for Example:\n{prompt}\n{'-' * 50}")  # Debug prompt
    # Generate caption
    llm.loadModel()
    output = llm.generateResponse(
        prompt
    )

    # Extract the actual caption
    caption_start = output.find("Caption:") + len("Caption:")
    caption = output[caption_start:].strip() if caption_start > 0 else output.strip()

    # Save results
    result = {"Input": data, "Caption": caption}

    # Print results for quick inspection
    print(f"Example:\nInput: {data}\nGenerated Caption: {caption}\n{'-' * 50}")
    return result


if __name__ == "__main__":
    import Models.LLAMA32
    # Run the test
    results = secondTest()
    #
    # Save results to a CSV for analysis
    df = pd.DataFrame(results)
    df.to_csv("generated_captions.csv", index=False)
    print("Results saved to 'generated_captions.csv'")
