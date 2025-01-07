from pylatexenc.latex2text import LatexNodes2Text
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification

import torch



class ALBERTDocRetrieval:

    def __init__(self, albert_tokenizer, albert_classifier):
        self.tokenizer = AlbertTokenizerFast.from_pretrained(albert_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(albert_classifier)


    def get_top_k_paragraphs(self, paragraphs, query, k, latex_to_plaintext=False):
        if k > len(paragraphs):
            k = len(paragraphs)
            #raise ValueError(f"k ({k}) cannot be greater than the number of paragraphs ({len(paragraphs)}).")
        

        # Turn latex into plaintext first
        if(latex_to_plaintext):
            query = LatexNodes2Text(math_mode='verbatim', strict_latex_spaces=True).latex_to_text(query)
            for i in range(len(paragraphs)):
                paragraphs[i] = LatexNodes2Text(math_mode='verbatim', strict_latex_spaces=True).latex_to_text(paragraphs[i])


        results = []
        for id, paragraph in enumerate(paragraphs):
            inputs = self.tokenizer.encode_plus(query, paragraph, return_tensors="pt", truncation=True)

            with torch.no_grad():
                outputs = self.model(**inputs)
                score = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][1].item()

            results.append((id, score))

        results.sort(key=lambda x: x[1], reverse=True)

        top_k_results = results[:k]
        return top_k_results