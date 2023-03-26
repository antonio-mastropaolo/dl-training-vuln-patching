# How the Training Procedure Impacts the Performance of Deep Learning-based Vulnerability Patching

Deep learning models that generate solutions for patching vulnerability have been widely used. However, the challenge is the limited availability of large datasets of patches for these models to learn from. To address this, researchers suggest starting with pre-trained models that possess general knowledge of programming languages or related tasks like bug fixing (See CodeT5 and VRepair). An alternative method, prompt-tuning, is also suggested to enhance fine-tuning instances to better exploit the pre-training knowledge. Despite the many attempts to automate vulnerability patching, there are no studies exploring the impact of different training methods on deep learning models' performance in this area. In this work we aim to fill this gap by comparing self-supervised and supervised pre-training solutions and experimenting with different prompt-tuning approaches. 

#### Pipeline Description

To replicate the experiments you can rely on this two files *vulrepair_main.py* and *vulrepair_main_prompt.py*.
While the former can be used to retrain the original VulRepair approach as well as the ablation model (i.e., T5-base without pre-training), the latter serves the promp-tuning procedure.  Before starting replicating any of the experiments we performed, make sure to install the requirements (see *requirements.txt*)

#### Data Statistics

Data statistics of the datasets are shown in the below table:

|         | #Examples   | #Examples       | #Examples
| ------- | :-------:   | :-------:       | :-------:
|         |   VulRepair |  VulRepair-WET  | VulRepair-FC
|  Train  |   5,937     |    5,913        |  4,905
|  Valid  |   839       |    836          |  836
|   Test  |   1,706     |    1,695        |  1,695

#### Fine-tuning  

*The following will be starting a fine-tuning procedure using the classic supervised approach*

##### Training

```shell
python3.9 vulrepair_main.py 
        --model_name=model.bin   
        --output_dir=./models/code-t5-self-supervised-classic-ft
        --tokenizer_name=Salesforce/codet5-base
        --model_name_or_path=Salesforce/codet5-base
        --do_train
        --epochs 75
        --encoder_block_size 512
        --decoder_block_size 256
        --train_batch_size 8     
        --eval_batch_size 8   
        --learning_rate 2e-5     
        --max_grad_norm 1.0     
        --evaluate_during_training     
        --seed 123456  2>&1 | tee ./models/code-t5-self-supervised-classic-ft/train.log

```

##### Inference

```shell
python3.9 vulrepair_main.py 
        --model_name=model.bin   
        --output_dir=./models/code-t5-self-supervised-classic-ft
        --tokenizer_name=Salesforce/codet5-base
        --model_name_or_path=Salesforce/codet5-base
        --do_test
        --encoder_block_size 512
        --decoder_block_size 256
        --eval_batch_size 8   
```

*To prompt-tune instead, run the following*

##### Training

```shell
python3.9 vulrepair_main_prompt.py 
        --model_name=model.bin   
        --output_dir=./models/prompt-tuning/soft-prompt/V1
        --tokenizer_name=Salesforce/codet5-base
        --model_name_or_path=Salesforce/codet5-base
        --do_train
        --epochs 75
        --encoder_block_size 512
        --decoder_block_size 256
        --train_batch_size 8     
        --eval_batch_size 8   
        --learning_rate 2e-5     
        --max_grad_norm 1.0     
        --evaluate_during_training    
        --soft_prompt     #this parameter can be factored out when running prompt-tuning using hard/discrete prompt templates
        --prompt_number=1 #change this parameter to experiment with a different templates {1-5} 
        --seed 123456  2>&1 | tee ./models/prompt-tuning/soft-prompt/V1/train.log

```

##### Inference

```shell
python3.9 vulrepair_main_prompt.py 
        --model_name=model.bin   
        --output_dir=../models/prompt-tuning/soft-prompt/V1
        --tokenizer_name=Salesforce/codet5-base
        --model_name_or_path=Salesforce/codet5-base
        --do_test
        --prompt_number=1  #change this parameter to experiment with a different templates {1-5}
        --soft_prompt      #this parameter can be factored out when running prompt-tuning using hard/discrete prompt templates
        --encoder_block_size 512
        --decoder_block_size 256
        --eval_batch_size 8   
```


#### Datasets :paperclip:

* The datasets for fine-tuning the models are stored on GDrive <a href="https://drive.google.com/drive/folders/1-3eLMTVLx8evwC9ROUBq9q-IdYHuy_WK?usp=sharing">here</a>
* The dataset for supervised pre-training (Chen et al.) is available here <a href="https://drive.google.com/drive/folders/1DtaNb2FaxGiei8DI1NGy4X9tYlYsxtp3?usp=sharing">here</a>

#### Models :bar_chart:
* <a href="https://drive.google.com/drive/folders/1pU0iDzd9sEigLKNlefsS2GytYRA1kvRo?usp=sharing">T5-base No-Pretraining (M0)</a>
* <a href="https://drive.google.com/drive/folders/1H_E_RI6ejOllCSSuuVh5ClCau1T9Mpdy?usp=sharing">VulRepair Replica (M1)</a>
* <a href="https://drive.google.com/drive/folders/1F6MQSLpN9ZRPrE3HX91XmakeJY1-lFw3?usp=sharing">Self-supervised + Fine-tuning on Bug-Fixing</a>
* <a href="https://drive.google.com/drive/folders/1D7Y3xkZ6RqzAEZ9HjLfZgM0AoOpDWUNg?usp=sharing">Self-supervised + Fine-tuning on Bug-Fixing + Fine-tuning on vulnerability (M2)</a>
* <a href="https://drive.google.com/drive/folders/1CkVunUmOJP6gVjr7srppWEnCF8mIY8sc?usp=sharing">Self-supervised + prompt fine-tuning on vulnerability (M3)</a>
* <a href="https://drive.google.com/drive/folders/14gvGIWMOKHIpM2wBaOWA0X_ewaBHVgsx?usp=sharing">Self-supervised + Fine-tuning on Bug-Fixing + prompt-tuning on vulnerability (M4)</a>



#### Results:  :open_file_folder:
* <a href="https://drive.google.com/drive/folders/1nQiWNcEJ9SUG1BhedvyD9T5lGvfN3hGC?usp=sharing">T5-base No-Pretraining (M0)</a>
* <a href="https://drive.google.com/drive/folders/1IAXWF0emc9JN0l762YoXwLxp9j2VZZM7?usp=sharing">VulRepair Replica (M1)</a>
* <a href="https://drive.google.com/drive/folders/1hMdlw76hDMa8Dy7ICgr6ZnqTLFkbjv-t?usp=sharing">Self-supervised + Fine-tuning on Bug-Fixing + Fine-tuning on vulnerability (M2)</a>
* <a href="https://drive.google.com/drive/folders/1YBGVqtoZcnrF_--LXbEQvWwX9wtIe9T_?usp=sharing">Self-supervised + prompt fine-tuning on vulnerability (M3)</a>
* <a href="https://drive.google.com/drive/folders/1i0f1SzhfFYybWt79LnGais83YMYYe_rC?usp=sharing">Self-supervised + Fine-tuning on Bug-Fixing + prompt-tuning on vulnerability (M4)</a>


#### Results:  :open_file_folder:

* <a href="https://drive.google.com/drive/folders/1q5xqMNw3boRGqnSq5Lk0dAbjbtrSyQzM?usp=sharing">Data for Statistical Tests (i.e., McNemar and Wilcoxon)</a>
