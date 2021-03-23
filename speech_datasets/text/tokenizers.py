from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Union

import sentencepiece as spm
from typeguard import check_argument_types


class AbsTokenizer(ABC):
    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def ids2tokens(self, ids: Iterable[int]) -> List[str]:
        raise NotImplementedError

    def text2ids(self, line: str) -> List[int]:
        return self.tokens2ids(self.text2tokens(line))

    def ids2text(self, ids: Iterable[int]) -> str:
        return self.tokens2text(self.ids2tokens(ids))

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class SentencepieceTokenizer(AbsTokenizer):
    def __init__(self, model: Union[Path, str],
                 token_list: Union[Path, str, Iterable[str]] = None):
        assert check_argument_types()
        self.model = str(model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model)

        if isinstance(token_list, (Path, str)):
            char_list = Path(token_list)
            with char_list.open("r", encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
        elif token_list is None:
            token_list = [self.sp.IdToPiece(i)
                          for i in range(self.sp.get_piece_size())]

        self.idx2tok = {i: tok for i, tok in enumerate(token_list)}
        self.tok2idx = {tok: i for i, tok in enumerate(token_list)}

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        return self.sp.EncodeAsPieces(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.sp.DecodePieces(list(tokens))

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.tok2idx.get(tok, self.tok2idx["<unk>"]) for tok in tokens]

    def ids2tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.idx2tok[idx] for idx in ids]

    def __len__(self):
        if self.idx2tok is None:
            return self.sp.get_piece_size()
        else:
            return len(self.idx2tok)
