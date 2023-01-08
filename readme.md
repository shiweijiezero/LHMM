This is a partial implementation for paper <u>LHMM: A Learning Enhanced HMM Model for Cellular Trajectory Map-matching</u>

---

### Overview

1. [Requirements](#requirements)
2. [Execution](#execution)
3. [Map-matching-Example](#Map-matching Example)
4. [Dataset](#Dataset)
5. [License](#license)
6. [Contact](#contact)

---

## 1. Requirements

The following modules are required.

- Ubuntu 16.04
- Python >=3.5 (`Anaconda3` recommended)
- PyTorch 0.4 (`virtualenv` recommended)
- Cuda 9.0

---

## 2. Execution

### 2.1 HMM Map-matching
Due to privacy protocols, we cannot render the Map-matching part. We try our best to provide the interface of the learned observation and transition probabilities. Moreover, a simple HMM framework is also provided in `src/pymatch.py`.


### 2.2 Training

```bash
$ cd pygcn
$ python gcn_train.py
$ cd trans
$ python trans_train.py
```

### 2.3 Serve

```bash
$ python gcnServer_old.py
$ python transServer.py
```
---

## 3. Map-matching-Example

We provide some figures to illustrate the matching of LHMM in director `Mapmatching-Figures`.

---

## 4. Dataset

We used real trajectory data. Unfortunately, due to privacy protection, we cannot provide the dataset for testing.

---

## 5. License

The code is developed under the MPL-02.0 license.

---

## 6. Contact

This version is up to rebuttal, new changes will be updated here soon.
If you have any questions or require further clarification, please do not hesitate to send an email to us (E-mail address：shiweijie0311@foxmail.com)
