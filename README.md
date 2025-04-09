# CliniQ: A Multi-faceted Benchmark for Electronic Health Record Retrieval with Semantic Match Assessment
![teaser_00](https://github.com/user-attachments/assets/7b21a843-033f-4361-9b8f-d7ed3b59de49)

CliniQ is a the first publicly avaiable benchmark for Electronic Health Record (EHR) retrieval.
The benchmark is built on 1,000 discharge summaries from MIMIC-III, split into 16,550 chunks of 100-word length.
CliniQ focuses on the task of entity retrieval, and considers three types of most frequently searched entities: diseases, procedures, and drugs.
We collect 1,246 unique queries sourced from ICD codes and NDC drug codes in MIMIC.
As for relevance judgments, we provide over 77k chunk-level relevance judgments annotated by GPT-4o, which achieves a Cohen's Kappa coefficient of 0.985 with expert annotations.

CliniQ supports two real-world retrieval settings: (1) Single-Patient Retrieval (finding relevant chunks within a note) and (2) Multi-Patient Retrieval (searching across multiple patients).
More importantly, CliniQ is the first benchmark assessing different types of semantic match capacities, including synonyms, hyponyms, abbreviations, and implication matches.

For more details, please refer to [our paper](https://arxiv.org/abs/2502.06252).

## Dataset Usage

### Dataset Access and Construction

Due to the access requirement of MIMIC, we are not allowed to share the corpus of our benchmark directly. 
Rather, we provide the `hadm_id` of the 1,000 discharge summaries we used along with the preprocessing script so that every one is able to reproduce the dataset.

First, please acquire access to [MIMIC-III](https://physionet.org/content/mimiciii/1.4/).
Then, run `preprocess.py` to produce `corpus.jsonl` under the directory `benchmark`.

### Dataset Structure

CliniQ benchmark consists of three parts: corpus, queries, and qrels.
Furthermore, the queries and qrels are split into three parts each: disease, procedure, and drug, corresponding to different query types.

We adopt similar format as [BEIR](https://github.com/beir-cellar/beir), and the only difference lies in that the labels in qrels files are free-text match types (string, synonym, hyponym, abbreviation, and implication).
Therefore, we provide evaluation script below to fit our benchmark.


### Model Evaluation

In `eval.py`, we provide code for evaluating all dense retrievers tested in our paper.
To evaluate a new model, choose appropriate embedding function or write your own.
Pay extra attention to whether the model encodes queries and documents differently.

The output of this script includes detailed metrics of different query types and semantic match types, along with a overall score as reported in our paper.

## Leaderboard
We currently do not provide a leaderboard due to the multi-faceted nature of our benchmark.
However, we are more than glad to hear from any participants and share the latest news regarding the benchmark with any one interested in this area.
Please feel free to email at zhengyun21@mails.tsinghua.edu.cn


## Citation
```
@misc{zhao2025cliniqmultifacetedbenchmarkelectronic,
      title={CliniQ: A Multi-faceted Benchmark for Electronic Health Record Retrieval with Semantic Match Assessment}, 
      author={Zhengyun Zhao and Hongyi Yuan and Jingjing Liu and Haichao Chen and Huaiyuan Ying and Songchi Zhou and Yue Zhong and Sheng Yu},
      year={2025},
      eprint={2502.06252},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.06252}, 
}
```
