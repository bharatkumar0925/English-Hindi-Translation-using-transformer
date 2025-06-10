from rouge_score import rouge_scorer
from sacrebleu.metrics import TER
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

def compute_rouge_scores(y_true, y_pred):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    for ref, hyp in zip(y_true, y_pred):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    return (
        sum(rouge1_scores) / len(rouge1_scores),
        sum(rouge2_scores) / len(rouge2_scores),
        sum(rougeL_scores) / len(rougeL_scores)
    )

def compute_meteor_scores(y_true, y_pred):
    # Tokenize the sentences
    tokenized_scores = [
        meteor_score([ref.split()], hyp.split()) for ref, hyp in zip(y_true, y_pred)
    ]
    return sum(tokenized_scores) / len(tokenized_scores)

def compute_ter_scores(y_true, y_pred):
    ter = TER()
    score = ter.corpus_score(y_pred, [y_true])
    return score.score

def compute_bleu_scores(y_true, y_pred):
    # Compute BLEU scores for 2-gram, 3-gram, and 4-gram
    smoothing_function = SmoothingFunction().method4  # Use method 4 for smoothing

    bleu_2 = corpus_bleu([[sentence.split()] for sentence in y_true], [pred.split() for pred in y_pred], weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    bleu_3 = corpus_bleu([[sentence.split()] for sentence in y_true], [pred.split() for pred in y_pred], weights=(0, 1, 0, 0), smoothing_function=smoothing_function)
    bleu_4 = corpus_bleu([[sentence.split()] for sentence in y_true], [pred.split() for pred in y_pred], weights=(0, 0, 1, 0), smoothing_function=smoothing_function)

    return bleu_2, bleu_3, bleu_4

# Example: Evaluation Function
def evaluate_metrics(y_true, y_pred):
    bleu_2, bleu_3, bleu_4 = compute_bleu_scores(y_true, y_pred)
    rouge1, rouge2, rougeL = compute_rouge_scores(y_true, y_pred)
    meteor = compute_meteor_scores(y_true, y_pred)
    ter = compute_ter_scores(y_true, y_pred)

    print(f"BLEU-2: {bleu_2:.4f}, BLEU-3: {bleu_3:.4f}, BLEU-4: {bleu_4:.4f}")
    print(f"ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"TER: {ter:.4f}")

