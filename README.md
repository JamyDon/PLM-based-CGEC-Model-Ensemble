# PLM-based CGEC Model Ensemble

Source code of the ACL 2023 short paper **Are Pre-trained Language Models Useful for Model Ensemble in Chinese Grammatical Error Correction?**

## Get Started
- Put our code into the [ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT) directory provided by [MuCGEC](https://github.com/HillZhang1999/MuCGEC) in order to run the ensemble.
- Put the output of models you want to ensemble into anywhere as you want. All these files should be formatted as M2.
- Set the `SRC_FILE` variant in our shell scripts in `ensemble` to the path of the source text file (one original sentence each line).
- Write the path of your model outputs into a text file (one path each line) and set the `MODEL` variant in our shell scripts in `ensemble` to this text file so that our ensemble script can read your model outputs.
- Download BERT-base-Chinese, MacBERT-base-Chinese and GPT2-Chinese to `ensemble/PLM`.
- Set the `OUT_DIR` variant in our shell scripts in `ensemble` to the directory of the ensemble result you want to save.
- Run the scripts in `ensemble` to carry out the ensemble.

## Our Model Ensemble Approaches
![diagram](diagram.png)
- Sentence-level Ensemble
- Edit-level Ensemble
- Edit-combination Ensemble

Please refer to the Python codes in `ensemble` for details of our ensemble strategies.

## Statements and Acknowledgements
- Our code is built on the basis of the [ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT) scorer provided by [MuCGEC](https://github.com/HillZhang1999/MuCGEC).
- Our work is supported by the National Hi-Tech RD Program of China (No.2020AAA0106600), the National Natural Science Foundation of China (62076008) and the Key Project of Natural Science Foundation of China (61936012).

## Citation
If you find our work useful for your research, please cite our paper:

**Are Pre-trained Language Models Useful for Model Ensemble in Chinese Grammatical Error Correction?**