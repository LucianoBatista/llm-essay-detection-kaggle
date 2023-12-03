Here you'll find what each experiment on the repository does. Some experiments will contains ETL process, others will have model training, others just validating some ideas.

# Exp 01

In this process I'm creating a new dataset to use on the competition. This dataset will contain a lot of essays written by humans and AI generated as well. I'm doing this because the original dataset from the competition contains almost none ai generated essays, and also because the leaderboard (test data) is evaluated on essays that was written also by another 5 prompts not showed on the training data.

The best part is that those prompts are comming from a well known corpus, the PERSUADE corpus.

The data generated here will contains some columns that will help us to validate different datasets from the community, I'm putting a "kaggle_repo" flag, to indicate from where the data come from.

The data looks like:

```sh
shape: (54_691, 6)
┌───────┬───────────┬───────────────────────────────────┬───────────┬───────────────┬─────────────┐
│ id    ┆ prompt_id ┆ text                              ┆ generated ┆ model         ┆ kaggle_repo │
│ ---   ┆ ---       ┆ ---                               ┆ ---       ┆ ---           ┆ ---         │
│ u32   ┆ str       ┆ str                               ┆ i64       ┆ str           ┆ i64         │
╞═══════╪═══════════╪═══════════════════════════════════╪═══════════╪═══════════════╪═════════════╡
│ 0     ┆ 0         ┆ Advantages of Limiting Car Usage… ┆ 1         ┆ gpt-3.5-turbo ┆ 1           │
│ 1     ┆ 0         ┆ Advantages of Limiting Car Usage… ┆ 1         ┆ gpt-3.5-turbo ┆ 1           │
│ 2     ┆ 0         ┆ Limiting car usage has numerous … ┆ 1         ┆ gpt-3.5-turbo ┆ 1           │
│ 3     ┆ 0         ┆ The passages provided discuss th… ┆ 1         ┆ gpt-3.5-turbo ┆ 1           │
│ …     ┆ …         ┆ …                                 ┆ …         ┆ …             ┆ …           │
│ 54687 ┆ -1        ┆ Working alone, students do not h… ┆ 0         ┆ human         ┆ 9           │
│ 54688 ┆ -1        ┆ "A problem is a chance for you t… ┆ 0         ┆ human         ┆ 9           │
│ 54689 ┆ -1        ┆ Many people disagree with Albert… ┆ 0         ┆ human         ┆ 9           │
│ 54690 ┆ -1        ┆ Do you think that failure is the… ┆ 0         ┆ human         ┆ 9           │
└───────┴───────────┴───────────────────────────────────┴───────────┴───────────────┴─────────────┘
```

Filtering duplicates and human written text (just for now), we endup with a small dataset with 823 AI generated text.

I'm planning for the next iteration, to use this 823 plus the data from the competition to create our classifier.

# Exp 02

:work_in_progress:
