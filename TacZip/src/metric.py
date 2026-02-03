import re
import string
from collections import Counter
from typing import Dict, List
from fuzzywuzzy import fuzz
from rouge import Rouge


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def answer_cleansing_zero_shot(dataset_name, pred):
    pred = pred.strip()
    if dataset_name in ("piqa", "social_i_qa", "arc_easy", "arc_challenge"):
        pred = pred.replace("According", "")
        pred = pred.replace("Base", "")
        pred = pred.replace("Answer", "")
        pred = pred.replace("Doc 1", "")
        pred = pred.replace("Doc 2", "")
        pred = pred.replace("Doc 3", "")
        pred = pred.replace("Doc 4", "")
        pred = pred.replace("Doc 5", "")
        pred = pred.replace("Document 1", "")
        pred = pred.replace("Document 2", "")
        pred = pred.replace("Document 3", "")
        pred = pred.replace("Document 4", "")
        pred = pred.replace("Document 5", "")

    if dataset_name in ("piqa"):
        pred = re.findall(r'A|B', pred)
    elif dataset_name in ("social_i_qa"):
        pred = re.findall(r'A|B|C', pred)
    elif dataset_name in("arc_easy", "arc_challenge"):
        pred = re.findall(r'A|B|C|D|E|1|2|3|4', pred)
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        # choose the first element in list ...
        pred = pred[0]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred


class Metric:
    @classmethod
    def compute(
        cls,
        predictions: List[str],
        answers: List[List[str]],
        metric_list: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        metric_list = [metric.lower() for metric in metric_list]
        cls._check_metric_list(metric_list)

        result = {}
        for metric in metric_list:
            total_score = 0
            for idx, (prediction, ground_truths) in enumerate(
                zip(predictions, answers)
            ):
                score = 0
                for ground_truth in ground_truths:
                    score = max(
                        score,
                        getattr(cls, metric)(
                            prediction,
                            ground_truth,
                            all_classes=kwargs["all_classes"][idx],
                        ),
                    )
                total_score += score
            result[metric] = total_score / len(predictions)

        return result

    @staticmethod
    def _check_metric_list(metric_list: List[str]):
        for metric in metric_list:
            assert hasattr(Metric, metric), f"Not find metric `{metric}`."

    @staticmethod
    def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
        rouge = Rouge()
        try:
            scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        except:
            return 0.0
        return scores["rouge-l"]["f"]

    @staticmethod
    def f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        return Metric.f1_score(prediction_tokens, ground_truth_tokens)

    @staticmethod
    def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
        em_match_list = []
        all_classes = kwargs["all_classes"]
        for class_name in all_classes:
            if class_name.lower() in prediction.lower():
                em_match_list.append(class_name.lower())
        for match_term in em_match_list:
            if match_term in ground_truth.lower() and match_term != ground_truth.lower():
                em_match_list.remove(match_term)
        if ground_truth.lower() in em_match_list:
            score = 1.0 / len(em_match_list)
        else:
            score = 0.0
        return score

    @staticmethod
    def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
        pattern = r"Paragraph (\d+)"
        matches = re.findall(pattern, ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth_id):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)

    @staticmethod
    def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
        numbers = re.findall(r"\d+", prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)

    @staticmethod
    def code_edit_sim(prediction: str, ground_truth: str, **kwargs) -> float:
        all_lines = prediction.lstrip("\n").split("\n")
        prediction = ""
        for line in all_lines:
            if ("`" not in line) and ("#" not in line) and ("//" not in line):
                prediction = line
                break
        return fuzz.ratio(prediction, ground_truth) / 100
    
    @staticmethod
    def em_score(prediction: str, ground_truth: str, **kwargs) -> float:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction == normalized_ground_truth:
            return 1.0
        else:
            return 0.0

