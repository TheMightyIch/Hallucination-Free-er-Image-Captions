# Function to create a prompt
import json

import pandas
import pandas as pd
import torch.cuda
from PIL import Image
import os
import importlib
import inspect

from pandas import DataFrame

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


def create_prompt_data(images: list[Image], model_list: list)-> pandas.DataFrame:
    result={("id", ""): [image.filename for image in images]}
    tagGenerationModel= [model for model in model_list if model.config["GenerationType"] == "Scene context"]
    for model in tagGenerationModel:
        print("Loading model: ", model.model_alias)
        model.loadModel()
        for image in images:
            result.setdefault((model.config["GenerationType"],model.model_alias), []).append(model.generateResponse(image))
        model.cleanModel()
        torch.cuda.empty_cache()
    otherModels=set(model_list)-set(tagGenerationModel)
    for model in otherModels:
        print("Loading model: ", model.model_alias)
        model.loadModel()
        for index, image in enumerate(images):
            if model.config["tag_input_needed"] == "yes":
                tags=[value[index] for key, value in result.items() if isinstance(key, tuple) and "Scene context" in key]
                result.setdefault((model.config["GenerationType"],model.model_alias), []).append(model.generateResponse(image, tags))
            else:
                result.setdefault((model.config["GenerationType"],model.model_alias), []).append(model.generateResponse(image))
        model.cleanModel()
        torch.cuda.empty_cache()
    df = pd.DataFrame(result)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def create_prompts(prompt:str, data: pandas.DataFrame)-> DataFrame:
    def create_prompt(row)->str:
        return prompt.format(Objects=row['Detected objects'].values,
           Context=str(row['Scene context'].values),
           OCR=row['OCR text'] if 'OCR text' in data.keys() else 'None',
            WorldKnowledge = row['WorldKnowledge'] if 'WorldKnowledge' in data.keys() else 'None')
    data["prompt"]=data.apply(create_prompt, axis=1)
    return data

def load_run_cfg(run_setting :str):
    file_name = f'{os.getcwd()}/Models/configs/{run_setting}.json'
    with open(file_name, 'r', encoding='utf-8') as cfile:
        data = json.load(cfile)
    return data

def generate_caption(llm, data: pandas.DataFrame):
    results = []
    llm.loadModel()
    for idx, row in data.iterrows():
        # Create prompt
        prompt = str(row["prompt"].values[0])
        print(prompt)  # Debug prompt
        # Generate caption
        output = llm.generateResponse(
            prompt
        )
        print(output)
        # Extract the actual caption
        caption_start = output.find("Caption:") + len("Caption:")
        caption = output[caption_start:].strip() if caption_start > 0 else output.strip()
        results.append(caption)
    llm.cleanModel()
    torch.cuda.empty_cache()
    data["caption"]=caption
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
        prompt = create_prompts(data)
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


def secondTest()-> DataFrame:
    cfg= load_run_cfg("RunSettings")
    models = define_models(cfg["Models"])
    llm = [model for model in models if model.config["GenerationType"] == "LLM"]
    try:
        llm=(lambda x: x)(*llm)
    except Exception as e:
        print(f"Error loading LLM, either to few or to many LLMs found: {e}")
        exit()

    models[:] = (model for model in models if model.config["GenerationType"] != "LLM")
    if len(models)> len(set(models)):
        assert "Error: Duplicate models detected"
        exit()
    image = [Image.open("image/AMBER_2.jpg")]
    print("Image loaded")
    data = create_prompt_data(image, models)
    print("Prompt info created")
    data = create_prompts(cfg["Prompt"],data)
    # Generate caption
    result = generate_caption(llm, data)
    return result


if __name__ == "__main__":
    import Models.LLAMA32
    # Run the test
    result = secondTest()
    #
    # Save results to a CSV for analysis
    result.to_csv("generated_captions.csv", index=False)
    print("Results saved to 'generated_captions.csv'")
