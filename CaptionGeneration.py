# Function to create a prompt
import ast
import importlib
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pandas
import pandas as pd
import torch.cuda
from PIL import Image
from pandas import DataFrame
from tqdm import tqdm

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


def create_prompt_data(images: dict[int : Image], model_list: list, cfg):
    result={("id",""): images.keys()}
    df=pd.DataFrame(result)
    df.to_csv(cfg["data_file_path"], index=False, mode="w")
    result={}
    tagGenerationModel= [model for model in model_list if model.config["GenerationType"] == "Scene context"]
    for model in (pbar:=tqdm(tagGenerationModel, desc="Loading scene context models", total=len(tagGenerationModel), position=0, leave=True)):
        pbar.set_postfix_str(f"Loading model: {model.model_alias}" )
        model.loadModel()
        for ids,image in (pbar:=tqdm(images.items(), desc="Scene context from Images", total=len(images), position=0, leave=True)):
            pbar.set_postfix_str(f"Image: {os.path.basename(image.filename)}",)
            result.setdefault("id",[]).append(int(ids))
            result.setdefault((model.config["GenerationType"],model.model_alias), []).append(model.generateResponse(image))
            if len(result["id"])>=cfg["batch_size"]:
                update_data(result ,cfg, key=(model.config["GenerationType"],model.model_alias))
                del result
                result={}
        model.cleanModel()
        torch.cuda.empty_cache()
        update_data(result ,cfg, key=(model.config["GenerationType"],model.model_alias))
        del result
        result = {}

    print("Scene context info created")
    otherModels=set(model_list)-set(tagGenerationModel)
    for model in (pbar:=tqdm(otherModels, desc="Loading models", total=len(otherModels), position=0, leave=True)):
        pbar.set_postfix_str(f"Loading model: {model.model_alias}")
        model.loadModel()
        tag_batch=pd.read_csv(cfg["data_file_path"],skiprows=0,nrows=cfg["batch_size"])
        tag_batch = tag_batch.dropna()
        tag_batch.columns = [ast.literal_eval(col) if col.startswith("(") else col for col in tag_batch.columns]
        for ids,image in (pbar:=tqdm(images.items(), desc="Scene context from Images", total=len(images), position=0, leave=True)):
            pbar.set_postfix_str(f"Image: {os.path.basename(image.filename)}",)
            result.setdefault("id", []).append(int(ids))
            if model.config["tag_input_needed"]:
                tags =tag_batch.loc[tag_batch["id"] == int(ids), [col for col in tag_batch.loc[
                    tag_batch["id"] == int(ids)].columns if "Scene context" in col]].values
                if tags.shape!=(1,):
                    tags=np.concatenate(tags).ravel()
                tags= [item for sublist in map(ast.literal_eval, tags) for item in sublist]
                result.setdefault((model.config["GenerationType"],model.model_alias), []).append(model.generateResponse(image, tags))
            else:
                result.setdefault((model.config["GenerationType"],model.model_alias), []).append(model.generateResponse(image))
            if len(result["id"])>=cfg["batch_size"]:
                update_data(result ,cfg, key=(model.config["GenerationType"],model.model_alias))
                del result
                result={}
                if model.config["tag_input_needed"]:
                    tag_batch=pd.read_csv(cfg["data_file_path"],skiprows=int(int(ids)/cfg["batch_size"])*cfg["batch_size"],nrows=cfg["batch_size"])
                    tag_batch = tag_batch.dropna(how="all")
                    column_names= pd.read_csv(cfg["data_file_path"],skiprows=0,nrows=1)
                    tag_batch.columns = column_names.columns
                    tag_batch.columns = [ast.literal_eval(col) if col.startswith("(") else col for col in tag_batch.columns]

        model.cleanModel()
        torch.cuda.empty_cache()
        update_data(result, cfg, key=(model.config["GenerationType"], model.model_alias))
        del result
        result = {}
    df = pd.read_csv(cfg["data_file_path"])
    df = df.dropna()
    df.to_csv(cfg["data_file_path"], index=False, mode="w")



def update_data(data: dict, cfg, key: tuple):
    df = pd.read_csv(cfg["data_file_path"])
    df.columns = [ast.literal_eval(col) if col.startswith("(") else col for col in df.columns]
    preprocessed= pd.DataFrame(data)
    #preprocessed.columns = pd.MultiIndex.from_tuples(preprocessed.columns)
    if key not in df.columns:
        df = df.merge(preprocessed, on="id", how="left")
    else:
        df = df.set_index("id").combine_first(preprocessed.set_index("id")).reset_index()
    df.to_csv(cfg["data_file_path"], index=False, mode="w")
    print("Data saved")
    del df

def create_prompts(prompt:str, data: pandas.DataFrame)-> DataFrame:
    def create_prompt(row)->str:
        return prompt.format(Objects=", ".join([item for sublist in map(ast.literal_eval,row[[index for index in row.index if "Detected objects" in index]].values) for item in sublist]),
           Context=", ".join([item for sublist in map(ast.literal_eval, row[[index for index in row.index if "Scene context" in index]].values) for item in sublist]),
           OCR=row[[index for index in row.index if "OCR text" in index]].values if 'OCR text' in data.keys() else 'None',
            WorldKnowledge = [item for sublist in map(ast.literal_eval, row[[index for index in row.index if "WorldKnowledge" in index]].values) for item in sublist] )
    data["prompt"]=data.apply(create_prompt, axis=1)
    return data

def load_run_cfg(run_setting :str):
    file_name = f'{os.getcwd()}/Models/configs/{run_setting}.json'
    with open(file_name, 'r', encoding='utf-8') as cfile:
        data = json.load(cfile)
    return data

def generate_caption(llm, data: pandas.DataFrame,show_prompt:bool=False)-> pandas.DataFrame:
    results = []
    try:
        llm.loadModel()
    except torch.OutOfMemoryError:
        print("Out of memory error")
        return data
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return data
    for idx, row in (pbar:=tqdm(data.iterrows(), desc="Generating captions", total=len(data), position=0, leave=True)):
        pbar.set_postfix_str(f"AMBER_: {row['id']}")
        # Create prompt
        prompt = str(row["prompt"])
          # Debug prompt
        # Generate caption
        output = llm.generateResponse(
            prompt
        )
        if show_prompt:
            print(prompt)
            print(output)
        # Extract the actual caption
        caption_start = output.find("Caption:") + len("Caption:")
        caption = output[caption_start:].strip() if caption_start > 0 else output.strip()
        results.append(caption)
    llm.cleanModel()
    torch.cuda.empty_cache()
    data["caption"]=results
    return data

def load_images(image_path: str)-> dict[int : Image]:
    images = {}
    files=os.listdir(image_path)
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            images[file.split("_")[-1].split(".")[0]]= Image.open(os.path.join(image_path, file))
    return images

def buildEvalJson(data: pandas.DataFrame, file_path:str = "captions.json"):
    result=[]
    for idx, row in data.iterrows():
        result.append({"id": int(row["id"]), "response": row["caption"]})
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

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


def main():
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
    if cfg["existing_data"]:
        try:
            result= pd.read_csv(cfg["data_file_path"])
            print("Data loaded")
        except FileNotFoundError:
            print(f"no file found at: {cfg["data_file_path"]}")
            exit()
    else:
        image = load_images(cfg["ImageFolder"])
        print("Image loaded")
        create_prompt_data(image, models,cfg=cfg)
        print("Prompt info created")

    if not cfg["existing_prompt"]:
        df = pd.read_csv(cfg["data_file_path"])
        df = df.dropna()
        df.columns = [ast.literal_eval(col) if col.startswith("(") else col for col in df.columns]
        df=create_prompts(cfg["Prompt"],df)
        df.to_csv(cfg["data_file_path"], index=False, mode="w")
        print("Prompt created")
    # Generate caption
    if not cfg["existing_data"]:
        del image
    if cfg["skip_caption_generation"]:
        pass
    else:

        df = pd.read_csv(cfg["data_file_path"])
        df = df.dropna()
        df.columns = [ast.literal_eval(col) if col.startswith("(") else col for col in df.columns]
        if not cfg["existing_caption"]:
            df=generate_caption(llm, df, cfg["show_prompt"] if "show_prompt" in cfg.keys() else False)
            df.to_csv(cfg["data_file_path"], index=False, mode="w")
        print("Caption loaded")

    if cfg["skip_evaluation"]:
        return
    df = pd.read_csv(cfg["data_file_path"])
    buildEvalJson(df, cfg["caption_file_storage"])
    command = [sys.executable, "AMBER/inference.py", "--inference_data", f"{cfg["caption_file_storage"]}", "--evaluation_type", "g", "--word_association", "AMBER/data/relation.json", "--safe_words","AMBER/data/safe_words.txt", "--annotation","AMBER/data/annotations.json", "--metrics", "AMBER/data/metrics.txt"]
    scores=subprocess.run(command,capture_output=True, text=True)
    df["score"]=scores.stdout
    print(scores.stdout)
    df.to_csv(cfg["result_path"]+cfg["result_name"], index=False, mode="w")
    print("Results saved to ")
    return


if __name__ == "__main__":
    # Run the test
    main()


