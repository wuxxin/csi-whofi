# Evaluation Metrics Explained

This document explains the evaluation metrics used in this project to assess the performance of the person re-identification model. The primary metrics are **Rank-k Accuracy** and **mean Average Precision (mAP)**.

## Core Concepts

Before diving into the specific metrics, let's define a few terms:

- **Query Image**: An image of a person we want to identify.
- **Gallery**: A collection of images of known individuals. For each person, there is one or more "ground truth" image in the gallery.
- **Similarity Score**: The model computes a similarity score between the query image and every image in the gallery. A higher score indicates a higher likelihood of being the same person.
- **Ranked List**: For each query, the gallery images are sorted in descending order based on their similarity scores. The top image is Rank-1, the second is Rank-2, and so on.

## Rank-k Accuracy

**Rank-k accuracy** measures whether the correct person is identified within the top `k` results of the ranked list.

- **Rank-1 Accuracy**: This is the percentage of queries where the highest-ranked gallery image is the correct one. It's the most straightforward measure of accuracy: "Did the model get it right on the first try?"
- **Rank-3 Accuracy**: This is the percentage of queries where the correct gallery image appears within the top 3 results.
- **Rank-5 Accuracy**: This is the percentage of queries where the correct gallery image appears within the top 5 results.

A higher Rank-k accuracy indicates a better-performing model. We track ranks 1, 3, and 5 to understand how often the correct person is "close" to the top, even if not the very first match.

## Mean Average Precision (mAP)

**Mean Average Precision (mAP)** provides a more comprehensive evaluation of the model's ranking performance across all queries.

To understand mAP, we first need to understand Average Precision (AP) for a single query. In the context of this project, there is only **one correct gallery image** for each query person (a single-ground-truth scenario).

For a single query, the **Average Precision (AP)** is calculated as:

```
AP = 1 / rank
```

where `rank` is the position of the correct gallery image in the ranked list.

- If the correct image is at Rank-1, the AP is 1/1 = 1.0.
- If the correct image is at Rank-2, the AP is 1/2 = 0.5.
- If the correct image is at Rank-10, the AP is 1/10 = 0.1.

The **mAP** is simply the average of the AP scores for all query images.

```
mAP = (Sum of AP for all queries) / (Number of queries)
```

Unlike Rank-k accuracy, which is a binary "success-or-failure" measure within the top `k` results, mAP rewards models that place the correct match higher in the ranking and penalizes those that place it lower, providing a more nuanced score of overall performance.
