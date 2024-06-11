# Adversarial Attack Tool using ART

This tool performs adversarial attacks on machine learning models deployed on servers, such as large language models (LLMs) or other models accessible via an API. It uses the Adversarial Robustness Toolbox (ART) to generate adversarial examples and evaluate the model's robustness.

## Prerequisites

- Python 3.6 or later
- Required Python libraries:
  - TensorFlow
  - Adversarial Robustness Toolbox (ART)
  - Requests
  - NumPy

You can install the necessary libraries using pip:

```shell
pip install tensorflow adversarial-robustness-toolbox requests numpy
```

## Usage
- Save the Script: Save the provided Python script (adversarial_attack_tool_llm.py) on your local machine.

- Prepare Your Model API: Ensure your model is deployed and accessible via an API endpoint. The API should accept JSON input data and return predictions in JSON format.

- Run the Script: Open a terminal or command prompt and navigate to the directory containing the script. Use the following command to run the script:

```shell
python adversarial_attack_tool_llm.py --api_url <API_URL> --attack_method fgsm --eps 0.2
```
- Replace <API_URL> with the actual URL of your deployed model’s API.

## Script Parameters
- --api_url: (required) API URL of the deployed model.
- -- attack_method: (optional, default: fgsm) Attack method to use (fgsm, pgd, zoo, boundary).
- --eps: (optional, default: 0.2) Attack perturbation strength.

## Notes for LLMs

- Input Format: Adjust the input data format in the model_query function to match the expected format of your LLM's API.
- Output Format: Ensure the API's response is parsed correctly to extract the predictions.

## Acknowledgements
This tool is powered by the Adversarial Robustness Toolbox (ART).
Developed by CryNoMor3.