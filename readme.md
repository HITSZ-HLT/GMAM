This is the code for emnlp2022 paper A Generative Model for End-to-End Argument Mining with Reconstructed Positional Encoding and Constrained Pointer Mechanism

**Task**: argument component identification (ACI) and argumentative relation identification (ARI).

Install the package in the requirements.txt

The evaluation dataset includes Argument Annotated Essays and Consumer Debt Collection Practices (CDCP).

## How to run

After enter the folder, you can run the code by directly using

```bash
python train.py --dataset essay
```

The performance is reported in average. You can check the performance in logs folder as follow:

```log
...
...
{"metric": {"Seq2SeqSpanMetric_essay_add_none": {"triple_f": 44.56, "triple_rec": 44.42, "triple_pre": 44.7, "am_component_f": 70.82000000000001, "am_component_rec": 71.43, "am_component_pre": 70.22, "em": 0.3333, "invalid": 0.0048, "entity_info": {"<<P>>": {"acc": 74.1784, "recall": 71.9818, "f1": 73.0636}, "<<C>>": {"acc": 62.1053, "recall": 67.0455, "f1": 64.4809}, "<<MC>>": {"acc": 68.75, "recall": 77.6471, "f1": 72.9282}}, "entity_overall": {"acc": 70.2247191011236, "recall": 71.42857142857143, "f1": 70.8215297450425}, "invalid_len": 0.0048, "invalid_order": 0.0, "invalid_cross": 0.0, "invalid_cover": 0.0}, "step": 1748, "epoch": 46}}
{"metric": {"data_test": {"Seq2SeqSpanMetric_essay_add_none": {"triple_f": 50.339999999999996, "triple_rec": 49.44, "triple_pre": 51.29, "am_component_f": 76.09, "am_component_rec": 75.85, "am_component_pre": 76.33, "em": 0.3593, "invalid": 0.0, "entity_info": {"<<P>>": {"acc": 81.1297, "recall": 78.1211, "f1": 79.597}, "<<C>>": {"acc": 65.1235, "recall": 70.0997, "f1": 67.52}, "<<MC>>": {"acc": 75.6579, "recall": 75.1634, "f1": 75.4098}}, "entity_overall": {"acc": 76.33466135458168, "recall": 75.85114806017418, "f1": 76.09213661636221}, "invalid_len": 0.0, "invalid_order": 0.0, "invalid_cross": 0.0, "invalid_cover": 0.0}}}}
```

## Check the results

We update the inference code to load the saved model, make inference and show the inferece results. You can see more details in "infer.py" . The infer_res.ipynb is provided to load the pkl file and convert the sequence results into readable strings.
