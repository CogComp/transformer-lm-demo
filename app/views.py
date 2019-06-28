import json
import os
import random

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from app.bert_lm import BERT_LM_predictions

# from bert_expansion import BertExpansion
from app.bert_expansion import BertExpansion

# BE = BertExpansion()
# from app.winograd import WinogradSolver

def winograd_frontend(request):
    return render(request, 'winograd.html', {})


# winoSolver = WinogradSolver()
BLM = BERT_LM_predictions()


# @csrf_exempt
# def winograd_solver(request, sent):
#     log = winoSolver.solve(sent, BLM)
#     predictedTokens = log["predictedTokens"]
#     mentions = log["mentions"]
#     table = []
#     random_value = random.choice(list(predictedTokens.values()))
#     num_rows = len(random_value)
#     for rowId in list(range(0, num_rows)):
#         row = []
#         for mentionId, _ in enumerate(mentions):
#             row.append(predictedTokens[mentionId][rowId])
#         table.append(row)
#
#     log["sent"] = sent
#     log["table"] = table
#
#     return render(request, 'winograd.html', log)

# ben_BLM = Ben_BERT_LM_predictions()

def bert_demo(request):
    context = {
        "alg": "perToken"
    }

    return render(request, 'bert.html', context)


@csrf_exempt
def bert_calculations(request, sent1, sent2, alg):
    ns_label = None
    ns_label_neg = None
    ns_label_score = None
    ns_label_neg_score = None
    ns_label_color = "red"
    if "@" not in sent2:
        # out = {"there is no masked token in the second sentence"}
        print("there is no masked token in the second sentence")
    if alg == "perToken":
        print("Pertoken calculations")
        predictedTokens, tokens = BLM.calculate_bert_masked_per_token(sent1, sent2)
        print("Tokens: " + str(tokens))
    elif alg == "greedy":
        predictedTokens, tokens = BLM.calculate_bert_masked_beam_search(sent1, sent2, beam_size=1)
    elif alg == "beamSearch":
        predictedTokens, tokens = BLM.calculate_bert_masked_beam_search(sent1, sent2, beam_size=3)
    elif alg == "nextSentence":
        ns_label, ns_label_neg, ns_label_score, ns_label_neg_score = BLM.calculate_next_sentence_prediction(sent1, sent2)
        predictedTokens, tokens = BLM.calculate_bert_masked_per_token(sent1, sent2)
        if ns_label == 1:
            ns_label = "False"
            ns_label_neg = "True"
        else:
            ns_label = "True"
            ns_label_neg = "False"
    if ns_label == "True":
        ns_label_color = "green"

    table = []
    random_value = random.choice(list(predictedTokens.values()))
    num_rows = len(random_value)
    for rowId in list(range(0, num_rows)):
        row = []
        for tokenId, _ in enumerate(tokens):
            # print(tokenId)
            if tokenId in predictedTokens:
                # print("yess .... ")
                row.append(predictedTokens[tokenId][rowId])
            else:
                row.append("")
        table.append(row)

    context = {
        "predictedTokens": predictedTokens,
        "tokens": tokens,
        "table": table,
        "sent1": sent1,
        "sent2": sent2,
        "alg": alg,
        "nextSentence": ns_label,
        "nextSentence_neg": ns_label_neg,
        "nextSentence_score": ns_label_score,
        "nextSentence_neg_score": ns_label_neg_score,
        "nextSentence_color": ns_label_color
    }
    return render(request, 'bert.html', context)
