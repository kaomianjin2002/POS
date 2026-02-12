from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
import re
from typing import Dict, Iterable, List, Sequence, Tuple

Sentence = List[Tuple[str, str]]


def _word_shape(word: str) -> str:
    if re.fullmatch(r"[0-9]+", word):
        return "NUM"
    if re.fullmatch(r"[A-Z]+", word):
        return "UPPER"
    if re.fullmatch(r"[A-Z][a-z]+", word):
        return "TITLE"
    if re.search(r"[0-9]", word):
        return "ALNUM"
    if re.fullmatch(r"[\u4e00-\u9fff]+", word):
        return "CJK"
    if re.fullmatch(r"\W+", word):
        return "PUNCT"
    return "OTHER"


class StructuredPerceptronTagger:
    def __init__(self, epochs: int = 8):
        self.epochs = epochs
        self.labels: List[str] = []
        self.weights: Dict[str, float] = defaultdict(float)
        self._avg_weights: Dict[str, float] = defaultdict(float)
        self._timestamps: Dict[str, int] = defaultdict(int)
        self._step = 0

    def _features(self, tokens: Sequence[str], i: int, prev_tag: str, tag: str) -> List[str]:
        token = tokens[i]
        feats = [
            f"bias::{tag}",
            f"w::{token}::{tag}",
            f"prev::{prev_tag}->{tag}",
            f"shape::{_word_shape(token)}::{tag}",
        ]
        if len(token) >= 1:
            feats.append(f"suf1::{token[-1:]}::{tag}")
            feats.append(f"pre1::{token[:1]}::{tag}")
        if len(token) >= 2:
            feats.append(f"suf2::{token[-2:]}::{tag}")
            feats.append(f"pre2::{token[:2]}::{tag}")
        return feats

    def _score(self, tokens: Sequence[str], i: int, prev_tag: str, tag: str) -> float:
        return sum(self.weights[f] for f in self._features(tokens, i, prev_tag, tag))

    def _update_feature(self, feat: str, delta: float):
        elapsed = self._step - self._timestamps[feat]
        self._avg_weights[feat] += elapsed * self.weights[feat]
        self._timestamps[feat] = self._step
        self.weights[feat] += delta

    def _update(self, tokens: Sequence[str], gold: Sequence[str], pred: Sequence[str]):
        prev_gold = "<s>"
        prev_pred = "<s>"
        for i, (g, p) in enumerate(zip(gold, pred)):
            if g != p:
                for feat in self._features(tokens, i, prev_gold, g):
                    self._update_feature(feat, 1.0)
                for feat in self._features(tokens, i, prev_pred, p):
                    self._update_feature(feat, -1.0)
            prev_gold, prev_pred = g, p
            self._step += 1

    def decode(self, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []
        dp: List[Dict[str, float]] = []
        back: List[Dict[str, str]] = []

        first_scores = {}
        first_back = {}
        for tag in self.labels:
            first_scores[tag] = self._score(tokens, 0, "<s>", tag)
            first_back[tag] = "<s>"
        dp.append(first_scores)
        back.append(first_back)

        for i in range(1, len(tokens)):
            curr_scores: Dict[str, float] = {}
            curr_back: Dict[str, str] = {}
            for tag in self.labels:
                best_prev = None
                best_score = -1e18
                for prev_tag in self.labels:
                    s = dp[i - 1][prev_tag] + self._score(tokens, i, prev_tag, tag)
                    if s > best_score:
                        best_score = s
                        best_prev = prev_tag
                curr_scores[tag] = best_score
                curr_back[tag] = best_prev or self.labels[0]
            dp.append(curr_scores)
            back.append(curr_back)

        last_tag = max(dp[-1], key=dp[-1].get)
        tags = [last_tag]
        for i in range(len(tokens) - 1, 0, -1):
            tags.append(back[i][tags[-1]])
        tags.reverse()
        return tags

    def fit(self, train_data: Sequence[Sentence]):
        label_set = set()
        for sent in train_data:
            for _, tag in sent:
                label_set.add(tag)
        self.labels = sorted(label_set)

        examples = [([w for w, _ in sent], [t for _, t in sent]) for sent in train_data]
        rng = random.Random(42)
        for _ in range(self.epochs):
            rng.shuffle(examples)
            for tokens, gold in examples:
                pred = self.decode(tokens)
                if pred != gold:
                    self._update(tokens, gold, pred)

        for feat in list(self.weights.keys()):
            elapsed = self._step - self._timestamps[feat]
            self._avg_weights[feat] += elapsed * self.weights[feat]
            self.weights[feat] = self._avg_weights[feat] / max(1, self._step)


class HMMTagger:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.labels: List[str] = []
        self.trans = defaultdict(Counter)
        self.emit = defaultdict(Counter)
        self.tag_counts = Counter()
        self.vocab = set()

    def fit(self, train_data: Sequence[Sentence]):
        labels = set()
        for sent in train_data:
            prev = "<s>"
            for word, tag in sent:
                labels.add(tag)
                self.vocab.add(word)
                self.trans[prev][tag] += 1
                self.emit[tag][word] += 1
                self.tag_counts[tag] += 1
                prev = tag
            self.trans[prev]["</s>"] += 1
        self.labels = sorted(labels)

    def _log_trans(self, prev: str, curr: str) -> float:
        denom = sum(self.trans[prev].values()) + self.alpha * (len(self.labels) + 1)
        num = self.trans[prev][curr] + self.alpha
        return math.log(num / denom)

    def _log_emit(self, tag: str, word: str) -> float:
        vocab_size = len(self.vocab) + 1
        denom = self.tag_counts[tag] + self.alpha * vocab_size
        num = self.emit[tag][word] + self.alpha
        return math.log(num / denom)

    def decode(self, tokens: Sequence[str]) -> List[str]:
        if not tokens:
            return []

        dp: List[Dict[str, float]] = []
        back: List[Dict[str, str]] = []

        first_scores = {}
        first_back = {}
        for tag in self.labels:
            first_scores[tag] = self._log_trans("<s>", tag) + self._log_emit(tag, tokens[0])
            first_back[tag] = "<s>"
        dp.append(first_scores)
        back.append(first_back)

        for i in range(1, len(tokens)):
            curr_scores = {}
            curr_back = {}
            for tag in self.labels:
                best_prev = None
                best_score = -1e18
                for prev_tag in self.labels:
                    score = dp[i - 1][prev_tag] + self._log_trans(prev_tag, tag) + self._log_emit(tag, tokens[i])
                    if score > best_score:
                        best_score = score
                        best_prev = prev_tag
                curr_scores[tag] = best_score
                curr_back[tag] = best_prev or self.labels[0]
            dp.append(curr_scores)
            back.append(curr_back)

        best_last = None
        best_score = -1e18
        for tag in self.labels:
            score = dp[-1][tag] + self._log_trans(tag, "</s>")
            if score > best_score:
                best_score = score
                best_last = tag

        tags = [best_last or self.labels[0]]
        for i in range(len(tokens) - 1, 0, -1):
            tags.append(back[i][tags[-1]])
        tags.reverse()
        return tags


def evaluate(model, test_data: Sequence[Sentence]) -> float:
    correct = 0
    total = 0
    for sent in test_data:
        tokens = [w for w, _ in sent]
        gold = [t for _, t in sent]
        pred = model.decode(tokens)
        for g, p in zip(gold, pred):
            correct += int(g == p)
            total += 1
    return correct / max(1, total)


@dataclass
class POSBundle:
    perceptron: StructuredPerceptronTagger
    hmm: HMMTagger
    metrics: Dict[str, float]


def split_data(data: Sequence[Sentence], ratio: float = 0.8) -> Tuple[List[Sentence], List[Sentence]]:
    n = max(1, int(len(data) * ratio))
    return list(data[:n]), list(data[n:])


def tokenize(text: str, lang: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if " " in text:
        return [tok for tok in text.split() if tok]
    if lang == "zh":
        return [ch for ch in text if not ch.isspace()]
    return re.findall(r"[A-Za-z]+|[0-9]+|[^\w\s]", text)


def english_corpus() -> List[Sentence]:
    return [
        [("I", "PRON"), ("love", "VERB"), ("NLP", "NOUN")],
        [("She", "PRON"), ("reads", "VERB"), ("books", "NOUN")],
        [("They", "PRON"), ("are", "AUX"), ("happy", "ADJ")],
        [("The", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
        [("A", "DET"), ("small", "ADJ"), ("dog", "NOUN"), ("barks", "VERB")],
        [("He", "PRON"), ("quickly", "ADV"), ("writes", "VERB")],
        [("We", "PRON"), ("study", "VERB"), ("machine", "NOUN"), ("learning", "NOUN")],
        [("Students", "NOUN"), ("submit", "VERB"), ("homework", "NOUN")],
        [("This", "DET"), ("course", "NOUN"), ("is", "AUX"), ("interesting", "ADJ")],
        [("Birds", "NOUN"), ("fly", "VERB"), ("high", "ADV")],
        [("My", "DET"), ("teacher", "NOUN"), ("explains", "VERB"), ("clearly", "ADV")],
        [("Coding", "NOUN"), ("improves", "VERB"), ("thinking", "NOUN")],
    ]


def chinese_corpus() -> List[Sentence]:
    return [
        [("我", "PRON"), ("爱", "VERB"), ("自然", "NOUN"), ("语言", "NOUN"), ("处理", "VERB")],
        [("她", "PRON"), ("喜欢", "VERB"), ("读", "VERB"), ("书", "NOUN")],
        [("今天", "NOUN"), ("天气", "NOUN"), ("很好", "ADJ")],
        [("学生", "NOUN"), ("认真", "ADV"), ("学习", "VERB")],
        [("老师", "NOUN"), ("讲解", "VERB"), ("清楚", "ADJ")],
        [("我们", "PRON"), ("正在", "AUX"), ("上", "VERB"), ("课", "NOUN")],
        [("这个", "DET"), ("方法", "NOUN"), ("非常", "ADV"), ("有效", "ADJ")],
        [("人工", "NOUN"), ("智能", "NOUN"), ("改变", "VERB"), ("世界", "NOUN")],
        [("他们", "PRON"), ("完成", "VERB"), ("作业", "NOUN")],
        [("模型", "NOUN"), ("训练", "VERB"), ("完成", "VERB")],
        [("数据", "NOUN"), ("质量", "NOUN"), ("重要", "ADJ")],
        [("系统", "NOUN"), ("运行", "VERB"), ("稳定", "ADJ")],
    ]


def train_bundle(lang: str) -> POSBundle:
    corpus = english_corpus() if lang == "en" else chinese_corpus()
    train, test = split_data(corpus)

    perceptron = StructuredPerceptronTagger(epochs=12)
    perceptron.fit(train)

    hmm = HMMTagger(alpha=0.2)
    hmm.fit(train)

    metrics = {
        "structured_perceptron": evaluate(perceptron, test),
        "hmm_baseline": evaluate(hmm, test),
    }
    return POSBundle(perceptron=perceptron, hmm=hmm, metrics=metrics)
