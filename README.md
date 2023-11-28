# NLP-Paper-Summarization
Summarize Long Document with Pretrained sequence-to-sequence LM with long-range attention! 


## How to use Model

# FLAN-T5-NLP-Paper-to-Question-Generation

This model is a fine-tuned version of [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) on an [allenai/QASPER: a dataset for question answering on scientific research papers ](https://huggingface.co/datasets/allenai/qasper)-based [NLP-Paper-to-QA-Generation](https://huggingface.co/datasets/UNIST-Eunchan/NLP-Paper-to-QA-Generation) dataset.

## Target Task

- NLP Paper's Abstract + Introduction --> {Question} [SEP] {Answer}
- Question-based Summarization
- Long Document Summarization
- Scientific Paper Summarization


## (1) How to use: Inference on CPU ( Code Snippets )
- Inference can be slow on CPU

### Load model directly 
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("UNIST-Eunchan/FLAN-T5-NLP-Paper-to-Question-Generation")
model = AutoModelForSeq2SeqLM.from_pretrained("UNIST-Eunchan/FLAN-T5-NLP-Paper-to-Question-Generation")
```

### Prompting Input
```python
txt =  r""" 
Generate Question, Answer pair correspond to the following research paper. 
[Abstract] + {text['abstract']} + [Introduction] + {text['introduction']}
Question, Answer:
""".replace("\n", "")

inputs = tokenizer(txt, max_length = 1024, truncation=True, padding="max_length", return_tensors="pt")
```

### For Multiple Question Generation  (👍)
```python
num_generate_sequence = 4 #8, 16, 2, 1
summaries = model.generate(input_ids =inputs["input_ids"], max_new_tokens=100, do_sample = True, top_p = 0.95, num_return_sequences = num_generate_sequence)
```
### For Single Question Generation   
```python
summaries = model.generate(input_ids =inputs["input_ids"], max_new_tokens=100, do_sample = True, top_p = 0.95)
```

```python
decoded_summaries = [tokenizer.decode(s, skip_special_tokens=False, clean_up_tokenization_spaces=True) for s in summaries]
decoded_summaries = [d.replace("<n>", " ").replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "") for d in decoded_summaries]

```

## (2) Faster Inference on GPU 
- about 60x faster than (1) [CPU --> COLAB T4 GPU]

### Additional Installation 
```python
!pip install accelerate -q
!pip install bitsandbytes -q
!pip install optimum -q
```

### Load model directly 
```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,BitsAndBytesConfig
from optimum.bettertransformer import BetterTransformer

# load model in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("UNIST-Eunchan/FLAN-T5-NLP-Paper-to-Question-Generation")
model = AutoModelForSeq2SeqLM.from_pretrained("UNIST-Eunchan/FLAN-T5-NLP-Paper-to-Question-Generation", quantization_config=quantization_config)
model = BetterTransformer.transform(model)
```


### For Multiple Question Generation  (👍)
```python
# use to(device)

num_generate_sequence = 16 # (about 20 sec with Colab T4 GPU)
summaries = model.generate(input_ids =inputs["input_ids"].to(device), max_new_tokens=100, do_sample = True, top_p = 0.95, num_return_sequences = num_generate_sequence)
```


### Training results


It achieves the following results on the evaluation set:
- Loss: 0.4504

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 0.99  | 46   | 34.6109         |
| 29.7732       | 1.99  | 92   | 16.5236         |
| 29.7732       | 2.98  | 138  | 4.6887          |
| 7.9911        | 3.97  | 184  | 0.5679          |
| 7.9911        | 4.97  | 230  | 0.4795          |
| 0.6152        | 5.96  | 276  | 0.4577          |
| 0.6152        | 6.95  | 322  | 0.4523          |
| 0.4811        | 7.95  | 368  | 0.4509          |
| 0.4811        | 8.94  | 414  | 0.4505          |
| 0.4721        | 9.93  | 460  | 0.4504          |

## Model description

- FLAN-T5-Large (783M) 



### Generated Output Example
- Our model generate 16 different Q-A Pair with top-p sampling.

```python
input: r""" 
Generate Question, Answer pair correspond to the following research paper. 
[Abstract] In this work, we explore prompt tuning, a simple yet effective mechanism for learning soft prompts to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's few-shot learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method closes the gap and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant in that large models are costly to share and serve, and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. Our method can be seen as a simplification of the recently proposed prefix tuning of Li and Liang (2021), and we provide a comparison to this and other similar approaches. Finally, we show that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer, as compared to full model tuning. [Introduction] With the wide success of pre-trained large language models, a range of techniques has arisen to adapt these general-purpose models to downstream tasks. ELMo (Peters et al., 2018) proposed freezing the pre-trained model and learning a task-specific weighting of its per-layer representations. However, since GPT (Radford et al., 2018) and BERT (Devlin et al., 2019), the dominant adaptation technique has been model tuning (or fine-tuning), where all model parameters are tuned during adaptation, as proposed by Howard and Ruder (2018).More recently, Brown et al. (2020) showed that prompt design (or priming) is surprisingly effective at modulating a frozen GPT-3 model’s behavior through text prompts. Prompts are typically composed of a task description and/or several canonical examples. This return to freezing pre-trained models is appealing, especially as model size continues to increase. Rather than requiring a separate copy of the model for each downstream task, a single generalist model can simultaneously serve many different tasks. Unfortunately, prompt-based adaptation has several key drawbacks. Task description is error-prone and requires human involvement, and the effectiveness of a prompt is limited by how much conditioning text can fit into the model’s input. As a result, downstream task quality still lags far behind that of tuned models. For instance, GPT-3 175B fewshot performance on SuperGLUE is 17.5 points below fine-tuned T5-XXL (Raffel et al., 2020) (71.8 vs. 89.3) despite using 16 times more parameters. Several efforts to automate prompt design have been recently proposed. Shin et al. (2020) propose a search algorithm over the discrete space of words, guided by the downstream application training data. While this technique outperforms manual prompt design, there is still a gap relative to model tuning. Li and Liang (2021) propose prefix tuning and show strong results on generative tasks. This method freezes the model parameters and backpropagates the error during tuning to prefix activations prepended to each layer in the encoder stack, including the input layer. Hambardzumyan et al. (2021) simplify this recipe by restricting the trainable parameters to the input and output subnetworks of a masked language model, and show reasonable results on classifications tasks. In this paper, we propose prompt tuning as a further simplification for adapting language models. We freeze the entire pre-trained model and only allow an additional k tunable tokens per downstream task to be prepended to the input text. This soft prompt is trained end-to-end and can condense the signal from a full labeled dataset, allowing our method to outperform few-shot prompts and close the quality gap with model tuning (Figure 1). At the same time, since a single pre-trained model is recycled for all downstream tasks, we retain the efficient serving benefits of frozen models (Figure 2). While we developed our method concurrently with Li and Liang (2021) and Hambardzumyan et al. (2021), we are the first to show that prompt tuning alone (with no intermediate-layer prefixes or task-specific output layers) is sufficient to be competitive with model tuning. Through detailed experiments in sections 2–3, we demonstrate that language model capacity is a key ingredient for these approaches to succeed. As Figure 1 shows, prompt tuning becomes more competitive with scale. We compare with similar approaches in Section 4. Explicitly separating task-specific parameters from the generalist parameters needed for general language-understanding has a range of additional benefits. We show in Section 5 that by capturing the task definition in the prompt while keeping the generalist parameters fixed, we are able to achieve better resilience to domain shifts. In Section 6, we show that prompt ensembling, learning multiple prompts for the same task, can boost quality and is more efficient than classic model ensembling. Finally, in Section 7, we investigate the interpretability of our learned soft prompts. In sum, our key contributions are: 1. Proposing prompt tuning and showing its competitiveness with model tuning in the regime of large language models. 2. Ablating many design choices, and showing quality and robustness improve with scale. 3. Showing prompt tuning outperforms model tuning on domain shift problems. 4. Proposing prompt ensembling and showing its effectiveness. 
Question, Answer:
""".replace("\n", "")

output= [' What was the size of each untrained model?[SEP] The size of the model can be a combination of the size of all the parameters in a model',
 ' What are the benefits of using soft prompts?[SEP] They reduce the need to use manual prompt design and conserve machine training data',
 ' What is the sample size of dataset?[SEP] 22840',
 ' How does the method outperform some of the pre-trained models?[SEP] They successfully tune their model for two tasks, one for a few shot and the other for several downstream tasks.',
 ' What is the sample size of the experiments?[SEP]135 for a simple task?[SEP]32 for a more complicated task',
 ' What is the baseline model they tested? [SEP] GPT-3 model, with four state-of-the-art examples in a masked language model',
 ' What task accuracy is given by prompts?[SEP]Mixed task efficiency was 93% and accuracy 85% compared to normal noise level',
 ' What metrics do they use?[SEP] EMO score, VSD, and SVM scores',
 ' What metrics are used to assess the performance of the soft prompt training?[SEP] quality of translation, accuracy of text-to-text, robustness of domain transfer, error rate.',
 ' How much do they experiment with the T5 baseline?[SEP] The baseline is used for simulated benchmarks.',
 ' Which task are they applying their method to?[SEP]They test their approach on classifications tasks',
 " Why do they show that their approach outperforms GPT-3's few-shot? [SEP] This is a large project that uses a multi-task approach to train GPT-3 models. In this paper, they demonstrate that the current method outperforms both the GPT-3 few-shot and the Li and Liang prefix tuning. They also show that the prefix tuning performed much better than the model tuning. What is the difference between their experiments",
 ' How do they compare with other techniques? [SEP] They provide a comparison for each approach.',
 ' Which task is the GPT-3 model most applicable to?[SEP]Classification tasks. For which tasks does the model need a subnetwork?[SEP]Classification tasks for GPT-3',
 ' What is the baseline test case used for this experiment?[SEP]Pompets for a variety of tasks are trained using the same method. This is the baseline, and the baseline is used for all applications.',
 ' What was the size of their model?[SEP] They experimented with 0.5 m.m and 0.5 m.m respectively.']

```


## Training and evaluation data
- Used Dataset: [UNIST-Eunchan/NLP-Paper-to-QA-Generation](https://huggingface.co/datasets/UNIST-Eunchan/NLP-Paper-to-QA-Generation) dataset.
- Train: dataset['train'] + dataset['test']
- Evaluation: dataset['validation'] 
 
### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 184
- num_epochs: 10



<!--
```python

MODEL_CARD = "UNIST-Eunchan/Pegasus-x-base-govreport-12288-1024-numepoch-10"

# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text2text-generation", model=MODEL_CARD)

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CARD)
```


## Trained Dataset 


평균 9000 토큰이 조금 안 되는 분량, 10000토큰 이상이 다수 포함된 U.S. Government Reoprt 문서 데이터를 원문, 500~1000 토큰 정도의 요약 문서를 요약 타깃으로 쌍으로 구성된 17,517개를 사용

This model-weight is based on abstractive summarization task-specific model (PEGASUS-X) and fine-tuned by using one NVIDIA RTX A6000 about 30 hours.

### INPUT 


Google Docs 기준 8 pages 정도 분량을 입력으로 받습니다.

책이나 A4지 기준으로 한 페이지에 500 words 정도 됩니다.



There are some similarities in how Medicare pays ASCs and hospital outpatient departments for the procedures they perform. However, the methods used by CMS to calculate the payment rates in each system, as well as the mechanisms used to revise the Medicare payment rates, differ. In 1980, legislation was enacted that enabled ASCs to bill Medicare for certain surgical procedures provided to Medicare beneficiaries. Under the ASC payment system, Medicare pays a predetermined, and generally all- inclusive, amount per procedure to the facility. The approximately 2,500 surgical procedures that ASCs may bill for under Medicare are assigned to one of nine payment groups that contain procedures with similar costs, but not necessarily clinical similarities. All procedures assigned to one payment group are paid at the same rate. Under the Medicare payment system, when more than one procedure is performed at the same time, the ASC receives a payment for each of the procedures. However, the procedure that has the highest payment rate receives 100 percent of the applicable payment, and each additional procedure receives 50 percent of the applicable payment. The Medicare payment for a procedure performed at an ASC is intended to cover the direct costs for a procedure, such as nursing and technician services, drugs, medical and surgical supplies and equipment, anesthesia materials, and diagnostic services (including imaging services), and the indirect costs associated with the procedure, including use of the facility and related administrative services. The ASC payment for a procedure does not include payment for implantable devices or prosthetics related to the procedure; ASCs may bill separately for those items. In addition, the payment to the ASC does not include payment for professional services associated with the procedure; the physician who performs the procedure and the anesthesiologist or anesthetist bill Medicare directly for their services. Finally, the ASC payment does not include payment for certain other services that are not directly related to performing the procedure and do not occur during the time that the procedure takes place, such as some laboratory, X-ray, and other diagnostic tests. Because these additional services are not ASC procedures, they may be performed by another provider. In those cases, Medicare makes payments to those providers for the additional services. For example, a laboratory service needed to evaluate a tissue sample removed during an ASC procedure is not included in the ASC payment. The provider that evaluated the tissue sample would bill and receive payment from Medicare for that service. Because ASCs receive one inclusive payment for the procedure performed and its associated services, such as drugs, they generally include on their Medicare claim only the procedure performed. In 1997, legislation was enacted that required the implementation of a prospective payment system for hospital outpatient departments; the OPPS was implemented in August 2000. Although ASCs perform only procedures, hospital outpatient departments provide a much broader array of services, including diagnostic services, such as X-rays and laboratory tests, and emergency room and clinic visits. Each of the approximately 5,500 services, including procedures, that hospital outpatient departments perform is assigned to one of over 800 APC groups with other services with clinical and cost similarities for payment under the OPPS. All services assigned to one APC group are paid the same rate. Similar to ASCs, when hospitals perform multiple procedures at the same time, they receive 100 percent of the applicable payment for the procedure that has the highest payment rate, and 50 percent of the applicable payment for each additional procedure, subject to certain exceptions. Like payments to ASCs, payment for a procedure under the OPPS is intended to cover the costs of the use of the facility, nursing and technician services, most drugs, medical and surgical supplies and equipment, anesthesia materials, and administrative costs. Medicare payment to a hospital for a procedure does not include professional services for physicians or other nonphysician practitioners. These services are paid for separately by Medicare. However, there are some differences between ASC and OPPS payments for procedures. Under the OPPS, hospital outpatient departments generally may not bill separately for implantable devices related to the procedure, but they may bill separately for additional services that are directly related to the procedure, such as certain drugs and diagnostic services, including X-rays. Hospital outpatient departments also may bill separately for additional services that are not directly related to the procedure and do not occur during the procedure, such as laboratory services to evaluate a tissue sample. Because they provide a broader array of services, and because CMS has encouraged hospitals to report all services provided during a procedure on their Medicare claims for rate-setting purposes, hospital claims may provide more detail about the services delivered during a procedure than ASC claims do. CMS set the initial 1982 ASC payment rates based on cost and charge data from 40 ASCs. At that time, there were about 125 ASCs in operation. Procedures were placed into four payment groups, and all procedures in a group were paid the same rate. When the ASC payment system was first established, federal law required CMS to review the payment rates periodically. In 1986, CMS conducted an ASC survey to gather cost and charge data. In 1990, using these data, CMS revised the payment rates and increased the number of payment groups to eight. A ninth payment group was established in 1991. These groups are still in use, although some procedures have been added to or deleted from the ASC-approved list. Although payments have not been revised using ASC cost data since 1990, the payment rates have been periodically updated for inflation. In 1994, Congress required that CMS conduct a survey of ASC costs no later than January 1, 1995, and thereafter every 5 years, to revise ASC payment rates. CMS conducted a survey in 1994 to collect ASC cost data. In 1998, CMS proposed revising ASC payment rates based on the 1994 survey data and assigned procedures performed at ASCs into payment groups that were comparable to the payment groups it was developing for the same procedures under the OPPS. However, CMS did not implement the proposal, and, as a result, the ASC payment system was not revised using the 1994 data. In 2003, MMA eliminated the requirement to conduct ASC surveys every 5 years and required CMS to implement a revised ASC payment system no later than January 1, 2008. During the course of our work, in August 2006, CMS published a proposed rule that would revise the ASC payment system effective January 1, 2008. In this proposed rule, CMS bases the revised ASC payment rates on the OPPS APC groups. However, the payment rates would be lower for ASCs. The initial OPPS payment rates, implemented in August 2000, were based on hospitals’ 1996 costs. To determine the OPPS payment rates, CMS first calculates each hospital’s cost for each service by multiplying the charge for that service by a cost-to-charge ratio computed from the hospital’s most recently reported data. After calculating the cost of each service for each hospital, the services are grouped by their APC assignment, and a median cost for each APC group is calculated from the median costs of all services assigned to it. Using the median cost, CMS assigns each APC group a weight based on its median cost relative to the median cost of all other APCs. To obtain a payment rate for each APC group, CMS multiplies the relative weight by a factor that converts it to a dollar amount. Beginning in 2002, as required by law, the APC group payment rates have been revised annually based on the latest charge and cost data. ... [(See more)](https://github.com/purang2/Report-Summarization/blob/main/data/GovReport%20Dataset.pdf)



### OUTPUT 

Medicare pays for surgical procedures performed at ambulatory surgical centers (ASC) and hospital outpatient departments through different payment systems. Although they perform a similar set of procedures, no comparison of ASC and hospital outpatient per-procedure costs has been conducted. The Medicare Prescription Drug, Improvement, and Modernization Act of 2003 directed GAO to compare the relative costs of procedures furnished in ASCs to the relative costs of those procedures furnished in hospital outpatient departments, in particular, how accurately the payment groups used in the hospital outpatient prospective payment system (OPPS) reflect the relative costs of procedures performed in ASCs. To do this, GAO collected data from ASCs through a survey. GAO also obtained hospital outpatient data from the Centers for Medicare & Medicaid Services (CMS). GAO determined that the payment groups in the OPPS, known as ambulatory payment classification (APC) groups, accurately reflect the relative cost of procedures performed in ASCs. GAO calculated the ratio between each procedure's ASC median cost, as determined by GAO's survey, and the median cost of each procedure's corresponding APC group under the OPPS, referred to as the ASC-to-APC cost ratio. GAO also compared the OPPS median costs of those same procedures with the median costs of their APC groups, referred to as the OPPS-to-APC cost ratio. GAO's analysis of the ASC-to-APC and OPPS-to-APC cost ratios showed that 45 percent of all procedures in the analysis fell within a 0.10 point range of the ASC-to-APC median cost ratio, and 33 percent of procedures fell within a 0.10 point range of the OPPS-to-APC median cost ratio. These similar patterns of distribution around the median show that the APC groups reflect the relative costs of procedures provided by ASCs as well as they reflect the relative costs of procedures provided in hospital outpatient departments and can be used as the basis for the ASC payment system. GAO's analysis also identified differences in the cost of procedures in the two settings. The median cost ratio among all ASC procedures was 0.39 and when weighted by Medicare claims volume was 0.84. The median cost ratio for OPPS procedures was 1.04. Thus, the cost of procedures in ASCs is substantially lower than the corresponding cost in hospital outpatient departments.
-->

