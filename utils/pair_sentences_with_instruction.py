from typing import List, Optional


def pair_sentences_with_instruction(sentences: List[str],
                                    instruction: Optional[str] = "Represent the query for retrieval") -> List[List[str]]:
    paired_sentences = [[sentence, instruction] for sentence in sentences]
    return paired_sentences
