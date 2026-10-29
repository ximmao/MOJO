import warnings
from collections.abc import Iterable
from typing import List, Union
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter


class InfiniteVocabEmbedding(nn.Module):
    def __init__(self, embedding_dim, init_scale=0.02):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.init_scale = init_scale
        self.padding_idx = 0

        self.weight = UninitializedParameter()
        self.vocab = None

        # Unfortunately, this hook is private, though there has been a PR to make it
        # public: https://github.com/pytorch/pytorch/issues/75287
        self._register_load_state_dict_pre_hook(
            self._hook_vocab_on_load_state_dict, with_module=False
        )

    def initialize_vocab(self, vocab: List[str]):
        assert self.vocab is None, (
            f"Vocabulary already initialized, and has {len(self.vocab)} words. "
        )
        "If you want to add new words to the vocabulary, use extend_vocab() instead."

        # Create a mapping from words to indices
        if isinstance(vocab, str):
            raise ValueError("vocab cannot be a single string")
        elif isinstance(vocab, Iterable):
            # check that all words are unique
            if len(vocab) != len(set(vocab)):
                raise ValueError("Vocabulary contains duplicate words")
            self.vocab = OrderedDict(zip(vocab, range(1, len(vocab) + 1)))
            assert "NA" not in self.vocab, "NA is a reserved word"
            self.vocab["NA"] = 0
            self.vocab.move_to_end("NA", last=False)
        else:
            raise ValueError("vocab must be a list or dict")

        self.initialize_parameters(len(self.vocab))

    def extend_vocab(self, vocab: List[str], exist_ok=False):
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        # find intersection and difference between key sets
        new_words, existing_words = [], []
        for word in vocab:
            if word not in self.vocab:
                new_words.append(word)
            else:
                existing_words.append(word)

        if not exist_ok and len(existing_words) > 0:
            raise ValueError(
                f"Vocabulary already contains {len(existing_words)} out of {len(vocab)}"
                f" words that are being added. You can skip this error by setting "
                f"exist_ok=True, but be aware that the embeddings for these existing "
                f"words won't be re-initialized."
            )

        # update tokenizer
        self.vocab.update(
            OrderedDict(
                zip(new_words, range(len(self.vocab), len(self.vocab) + len(new_words)))
            )
        )

        # check that the dictionary update was done correctly
        assert len(self.vocab) == len(self.weight) + len(new_words), (
            f"Expected vocabulary length to be {len(self.weight) + len(new_words)}, "
            f"but got {len(self.vocab)}"
        )

        # check that the largest value in the vocab matches its size
        assert max(self.vocab.values()) == len(self.vocab) - 1, (
            f"Expected maximum vocabulary index to be {len(self.vocab) - 1}, "
            f"but got {max(self.vocab.values())}"
        )

        # make a copy of existing embeddings
        embeddings_for_existing_words = self.weight.clone().detach()

        # reinitalize weight matrix after extending it
        self.weight = UninitializedParameter(device=self.weight.device)
        self.initialize_parameters(len(self.vocab))

        # copy existing embeddings into new weight matrix
        self.weight.data[: len(embeddings_for_existing_words)] = (
            embeddings_for_existing_words
        )
        return self

    def subset_vocab(self, vocab: List[str], inplace=True):
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        assert len(vocab) > 0, "Vocabulary must contain at least one word."

        # check that all words are unique
        if len(vocab) != len(set(vocab)):
            raise ValueError("Vocabulary contains duplicate words")

        # find intersection and difference between key sets
        word_indices = [0]  # NA will be added
        for word in vocab:
            if word not in self.vocab:
                raise ValueError(f"Vocabulary does not contain word {word}")
            else:
                word_indices.append(self.tokenizer(word))

        # make a copy of selected embeddings
        word_indices = torch.tensor(word_indices, dtype=torch.long)
        with torch.no_grad():
            embeddings_for_selected_words = self.weight[word_indices].clone().detach()

        if inplace:
            self.vocab = None
            self.weight = UninitializedParameter()

            self.initialize_vocab(vocab)

            self.weight.data = embeddings_for_selected_words
            return self
        else:
            new_embedding = self.__class__(self.embedding_dim, self.init_scale)
            new_embedding.initialize_vocab(vocab)
            new_embedding.weight.data = embeddings_for_selected_words
            return new_embedding

    def tokenizer(self, words: Union[str, List[str]]):
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        if isinstance(words, str):
            return self.vocab[words]
        return [self.vocab[w] for w in words]

    def detokenizer(self, index: int):
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")

        return list(self.vocab.keys())[index]

    def is_lazy(self):
        return isinstance(self.weight, UninitializedParameter)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module, but will not reset the
        vocabulary."""
        if not self.is_lazy():
            torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)

    def initialize_parameters(self, num_embeddings) -> None:  # type: ignore[override]
        if self.is_lazy():
            with torch.no_grad():
                self.weight.materialize((num_embeddings, self.embedding_dim))
                self.reset_parameters()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.is_lazy():
            destination[prefix + "weight"] = self.weight
            destination[prefix + "vocab"] = self.vocab.copy() if self.vocab else None
        else:
            super()._save_to_state_dict(destination, prefix, keep_vars)
            destination[prefix + "vocab"] = self.vocab.copy()

    def _hook_vocab_on_load_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not self.is_lazy():
            incoming_vocab = state_dict.pop(prefix + "vocab")

            remap_indices = []
            for word, index in self.vocab.items():
                if not word in incoming_vocab:
                    raise ValueError(
                        f"Vocabulary mismatch: word {word} is missing. If "
                        "you would like to add new words, or a new "
                        "vocabulary, do not initialize the vocab, load the"
                        " checkpoint, and then call extend_vocab() to add"
                        "new words, and/or subset_vocab() to remove words."
                    )
                remap_indices.append(incoming_vocab.pop(word))
            if len(incoming_vocab) > 0:
                raise ValueError(
                    f"Vocabulary mismatch: {len(incoming_vocab)} words are"
                    "remaining and cannot be loaded. If you would like to "
                    "load a subset of the vocabulary, do not initialize "
                    "the vocab, load the checkpoint, and then call "
                    "subset_vocab()"
                )
            remap_indices = torch.tensor(remap_indices, dtype=torch.long)

            state_dict[prefix + "weight"] = state_dict[prefix + "weight"][remap_indices]

        else:
            if not isinstance(state_dict[prefix + "weight"], UninitializedParameter):
                # The module is not initialized, but the one being loaded is
                with torch.no_grad():
                    self.weight.materialize(state_dict[prefix + "weight"].shape)
                self.vocab = state_dict.pop(prefix + "vocab")
            else:
                # Both the module and the one being loaded are not initialized
                assert state_dict.pop(prefix + "vocab") is None

    def forward(self, input):
        if self.is_lazy():
            raise ValueError("No vocabulary was initialized. Use initialize_vocab()")
        return F.embedding(input, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        return "embedding_dim={}, num_embeddings={}".format(
            self.embedding_dim, len(self.vocab) if self.vocab is not None else 0
        )
