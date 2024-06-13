from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.mixedbreadai_rerank import MixedbreadAIRerank


def test_class():
    names_of_base_classes = [b.__name__ for b in MixedbreadAIRerank.__mro__]
    assert BaseNodePostprocessor.__name__ in names_of_base_classes
