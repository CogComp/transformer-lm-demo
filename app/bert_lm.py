import math
from typing import List

import torch
from pytorch_pretrained_bert import BertForPreTraining, BertTokenizer


class BERT_LM_predictions:
    use_gpu = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    max_batch_size = 250  # max number of instanes grouped together
    batch_max_length = 10  # max number of tokens in each instance

    bert_model = 'bert-base-uncased'

    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        # Load pre-trained model (weights)
        self.model = BertForPreTraining.from_pretrained('bert-base-uncased')
        self.model.eval()

    def vectorize_maked_instance(self, tokenized_text1: List[str], tokenized_text2: List[str]):

        tokens = []
        segment_ids = []
        input_mask = []

        masked_indices = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        input_mask.append(1)

        for token in tokenized_text1:
            tokens.append(token)
            segment_ids.append(0)
            input_mask.append(1)

        tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)

        second_part_start_index = len(tokens)
        for token in tokenized_text2:
            if token == "@":
                masked_indices.append(len(tokens))
                tokens.append("[MASK]")
            else:
                tokens.append(token)
            segment_ids.append(1)
            input_mask.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)
        input_mask.append(1)
        token_length = len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return input_ids, segment_ids, input_mask, token_length, second_part_start_index, masked_indices, tokens

    def calculate_next_sentence_prediction(self, sent1, sent2):
        tokenized_text1 = self.tokenizer.tokenize(sent1)
        tokenized_text2 = self.tokenizer.tokenize(sent2)

        input_ids, segment_ids, input_mask, token_length, second_part_start_index, masked_indices, tokens = self.vectorize_maked_instance(
            tokenized_text1, tokenized_text2)
        input_tensor = torch.tensor([input_ids])
        segment_tensor = torch.tensor([segment_ids])
        mask_tensor = torch.tensor([input_mask])

        _, seq_score = self.model(input_tensor, segment_tensor, mask_tensor)

        win_idx = torch.argmax(seq_score, dim=1).item()
        lose_idx = 0
        if win_idx == 0:
            lose_idx = 1

        seq_score = seq_score.detach().numpy()

        return win_idx, lose_idx, round(seq_score[0][win_idx], 3), round(seq_score[0][lose_idx], 3)

    def calculate_bert_masked_per_token(self, sent1, sent2, k = 30):
        # print("calculating the representations . . . ")
        # print(self.cache.volume())
        tokenized_text1 = self.tokenizer.tokenize(sent1)
        tokenized_text2 = self.tokenizer.tokenize(sent2)

        # assert "@" in tokenized_text2, "there is no masken token in the 2nd sentence"

        input_ids, segment_ids, input_mask, token_length, second_part_start_index, masked_indices, tokens = self.vectorize_maked_instance(
            tokenized_text1, tokenized_text2)

        input_tensor = torch.tensor([input_ids])
        segment_tensor = torch.tensor([segment_ids])
        mask_tensor = torch.tensor([input_mask])

        # Predict all tokens
        predictions, _ = self.model(input_tensor, segment_tensor, mask_tensor)

        predictedTokens = {}

        import numpy as np

        # if len(masked_indices) == 0:
        masked_indices = list(np.arange(0, token_length))

        print("masked_indices: " + str(masked_indices))

        # calculating predictions
        for ind in masked_indices:
            top_scores, top_indices = torch.topk(predictions[0, ind], k)
            top_scores = top_scores.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
            predictedTokens[ind] = [(self.tokenizer.convert_ids_to_tokens([id])[0], normlalize(s)) for id, s in
                                    zip(top_indices, top_scores)]

        return predictedTokens, tokens

    def calculate_bert_masked_beam_search(self, sent1, sent2, beam_size):
        # print(self.cache.volume())
        tokens1 = self.tokenizer.tokenize(sent1)
        tokens2 = self.tokenizer.tokenize(sent2)

        predictedTokens = {}

        def beam_search(tokenized_text1, tokenized_text2, selected_scores, selected_tokens):
            # print("-------")
            input_ids, segment_ids, input_mask, token_length, second_part_start_index, masked_indices, tokens = self.vectorize_maked_instance(
                tokenized_text1, tokenized_text2)

            output_list = []
            if len(masked_indices) > 0:
                input_tensor = torch.tensor([input_ids])
                segment_tensor = torch.tensor([segment_ids])

                # Predict all tokens
                predictions, _ = self.model(input_tensor, segment_tensor)

                # calculating predictions
                ind = masked_indices[0]  # take the first index
                top_scores, top_indices = torch.topk(predictions[0, ind], 10)
                top_scores = top_scores.cpu().tolist()
                top_indices = top_indices.cpu().tolist()
                predicted = [(self.tokenizer.convert_ids_to_tokens([id])[0], normlalize(s)) for id, s in
                             zip(top_indices, top_scores)]

                # replace the first mask with top tokens:
                for token, score in predicted[0:beam_size]:
                    # print(token)
                    # print(masked_indices)
                    # print(ind)
                    # print(tokenized_text2)
                    tokenized_text2_new = tokenized_text2.copy()
                    tokenized_text2_new[ind - 2 - len(tokenized_text1)] = token  # 2 extra shift for [CLS] and [SEP]
                    # print(tokenized_text2_new)

                    selected_tokens_new = selected_tokens.copy()
                    selected_tokens_new.append(token)

                    selected_scores_new = selected_scores.copy()
                    selected_scores_new.append(score)

                    if len(masked_indices) > 1:
                        output_list.extend(
                            beam_search(tokenized_text1, tokenized_text2_new, selected_scores_new, selected_tokens_new))
                    else:
                        output_list.append((selected_scores_new, selected_tokens_new))
                # print("intermediate: " + str(output_list))
            return output_list

        predicted_sequences = beam_search(tokens1, tokens2, [], [])

        # sorted list
        predicted_sequences = sorted(predicted_sequences, key=lambda x: -sum(x[0]))
        # for score, tokens in predicted_sequences:

        input_ids, segment_ids, input_mask, token_length, second_part_start_index, masked_indices, tokens = self.vectorize_maked_instance(
            tokens1, tokens2)

        assert len(masked_indices) == len(predicted_sequences[0][0])

        for ind in masked_indices:
            predictedTokens[ind] = []
        for scores, sequence in predicted_sequences[0:10]:
            for i, s in enumerate(scores):
                t = sequence[i]
                index = masked_indices[i]
                predictedTokens[index].append((t, s))

        # selected the best sequences
        # print(predicted_sequences)
        # print(predictedTokens)

        return predictedTokens, tokens

    def set_expansion(self, seed_set):
        # sep = ", to "
        sep = ", "
        # sep = "; "
        # seed_set = [x.replace("_", " ") for x in seed_set]
        see_string = sep.join(seed_set)
        # before = "For example, to "
        # before = "( to "
        # before = "("
        before = "( For example, "
        # before = "( For instance, "
        after = ", etc)"
        # after = ")"
        # after = "."
        sent2 = f"{before + see_string + sep} @ {after}"
        # print(sent2)
        predictedTokens, _ = self.calculate_bert_masked_per_token("", sent2, k = 80)
        assert len(predictedTokens.keys()) == 1
        return list(predictedTokens.values())[0]

    def analogy(self, b, c):
        print("Looking for the ", b, " of the ", c)

        # query: (cleveland, @; @, texas)
        # sent2 = f"({b}, @; @, {c})."

        # query: (Indianapolis of @) is like (@ of Pennsylvania).
        # sent2 = f"({b} of @) is like (@ of {c})."

        # query: ( voldemort of @; @ of tolkien)
        sent2 = f"({b} of @; @ of {c})."

        predictedTokens, _ = self.calculate_bert_masked_per_token("", sent2, k=80)
        # print(predictedTokens.keys())
        assert len(predictedTokens.keys()) == 2
        return list(predictedTokens[max(predictedTokens.keys())])


def normlalize(number):
    return math.floor(number * 1000) / 1000.0


if __name__ == '__main__':
    BLM = BERT_LM_predictions()
    # output = BLM.calculate_bert_masked_per_token("", "Who was Jim Henson ? Jim @ was a puppeteer")
    # output = BLM.calculate_bert_masked_beam_search("abc", "@ and @ is located in USA. ", beam_size=2)
    # output = BLM.set_expansion(["Ford", "Honda"])
    # output = BLM.set_expansion(["Ford", "Nixon"])
    # output = BLM.set_expansion(["Ford", "Chevy"])
    # output = BLM.set_expansion(["Harrison Ford", "Depp"])
    # output = BLM.set_expansion(["Safari", "Trip"])
    output = BLM.set_expansion(["Safari", "I.e."])
    print(output)
