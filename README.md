
# Hallucination-Free-er-Image-Captions

This is a pipeline used for image caption generation. It uses several image experts to extract features and a LLM to generate the caption out of it and afterwards applying the **AMBER benchmark** to it, to have a Hallucination score. [[1]](#1)


## Installation

To install all required dependencies follow the steps below. 

```bash
  pip install -e
```
also install troch depending on your cuda version
```bash
  pip install torch==?
```
Download all images you want to run it with, e.g. the AMBER images and add them to the defined `ImageFolder:` of the `RunSettings`. They should all be in the `'RGB'` format

download the [ram_plus_swin_large_14m.pth](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) file from the [Ram++ repo](https://github.com/xinyu1205/recognize-anything) and add it to the Ram++ `model_id` location defined in the config file.
    
## Running Tests

To run tests, add the required information in the config files under 
`/Models/configs`. The `RunSettings.json` file contains all settings for the caption generation and benchmarking. In the `RunSettings` the `Models:` point contains a list of each model used in the pipeline. The usage of these models is defined in their own config files. To execute it either run: 
```bash
  python.exe CaptionGeneration.py
```
or the `if __name__ == "__main__":` in the file directly with an editor of your choice. 

### Adding new models
If a new model should be added to the pipeline the following points must be complied with:
- The new model should implement the `AbstractModel` class
- The filename and the classname should be the same
- the config file should follow the naming convention of `alias_class/filename.json`
- the config file has to contain:
    - `model_id:` the exact id of the model, either hugging face or third party id
    - `model_type:` the type of the model `["Scene Context", "Detected object", "LLM", "OCR text", "WorldKnowledge"]`
    - `ToDevice:` boolean if the model should be loadet `.to(device)` once it is used. 





## Authors
- [@TheMightyIch](https://github.com/TheMightyIch)
- [@ShrineethKotian](https://github.com/ShrineethKotian)
- [@aishwaryagrao](https://github.com/aishwaryagrao)
- [@naveenc18](https://github.com/naveenc18)



## Appendix

<a id="1">[1]</a> 
  title={An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation},
  author={Wang, Junyang and Wang, Yuhang and Xu, Guohai and Zhang, Jing and Gu, Yukai and Jia, Haitao and Yan, Ming and Zhang, Ji and Sang, Jitao},
  journal={arXiv preprint arXiv:2311.07397},
  year={2023}
