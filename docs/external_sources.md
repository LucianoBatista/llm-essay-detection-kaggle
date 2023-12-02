The objective here is to describe any external public available source of essay, that can be used on the competition.

Important note, there is a `persuade` corpus, this is public, and seems that the competition is using the 2 and 12 prompts from this corpus, to give to students to write an essay.

More info about it:

- [Here](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/452172)
- [And Here](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/453410)

# LLM Generated Essays for the Detect AI Comp

This dataset contains 700 LLM generated essays in total, where:

- 500 generated by the gpt-3.5-turbo
- 200 generated by the gpt-4

Those prompt ids, seems to be from the competition.

- [Dataset](https://www.kaggle.com/datasets/radek1/llm-generated-essays/data?select=ai_generated_train_essays_gpt-4.csv)

# ArguGPT

ArguGPT is a balanced corpus of 4,038 argumentative essays generated by 7 GPT models in response to essay prompts from three sources:

1. in-class or homework exercises
2. TOEFL
3. GRE writing tasks.

Those models was not identified, the paper says the following:

> _...seven models of the GPT family (GPT2-XL, variants of GPT3, and ChatGPT)_

One important observation is that the paper also evaluate human written text, but I couldn't find those texts.

I'll link to paper, very useful information there:

- [Paper](https://arxiv.org/abs/2304.07666)
- [Dataset](https://www.kaggle.com/datasets/alejopaullier/argugpt/?select=argugpt.csv)

# DAIGT | External Dataset

I'm using just the AI generated part of the dataset. This dataset also has a column of student written essays, but it is a bit confusing about how this is structure. This dataset was generated by using chatgpt.

- [Dataset](https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset/)

# ...