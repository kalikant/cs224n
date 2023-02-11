batch_size = 96
n_epochs = 2
base_LM_model = "roberta-base"
max_seq_len = 386
learning_rate = 3e-5
lr_schedule = "LinearWarmup"
warmup_proportion = 0.2
doc_stride = 128
max_query_length = 64

QASSA2017 = "http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html"