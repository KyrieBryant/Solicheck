# SoliCheck: Smart Contract Vulnerability Detection Framework

Smart contracts are crucial for blockchain ecosystems, but their vulnerabilities can cause huge losses.

## Shortcomings of Existing Tools

Traditional smart contract detectors rely on expert - made rules. They’re ineffective and can’t scale as the number of smart contracts grows. Neural - network - based methods show potential, yet a full understanding of smart contract features is missing.

## SoliCheck at a Glance

SoliCheck is our solution to these problems.

**Fusion Model**: It combines BiLSTM and CodeBERT. This helps capture sequential and semantic details in smart contract code.

**FAU Representation**: FAU integrates smart contract source code and its AST. So, SoliCheck analyzes both raw code and its structure.

## Evaluation Results

We tested SoliCheck on public datasets, comparing it with traditional and neural - network - based tools. SoliCheck achieved higher F1 scores than current top - performing methods. Ablation experiments also proved the effectiveness of its fusion model and code - understanding approach.

SoliCheck combines advanced techniques to offer a more accurate and scalable way to secure blockchain ecosystems.
