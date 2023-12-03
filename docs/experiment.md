Here you'll find what each experiment on the repository does. Some experiments will contains ETL process, others will have model training, others just validating some ideas.

# Exp 01

In this process I'm creating a new dataset to use on the competition. This dataset will contain a lot of essays written by humans and AI generated as well. I'm doing this because the original dataset from the competition contains almost none ai generated essays, and also because the leaderboard (test data) is evaluated on essays that was written also by another 5 prompts not showed on the training data.

The best part is that those prompts are comming from a well known corpus, the PERSUADE corpus.

The data generated here will contains some columns that will help us to validate different datasets from the community, I'm putting a "kaggle_repo" flag, to indicate from where the data come from.

The data looks like:
...
