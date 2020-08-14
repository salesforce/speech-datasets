from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Iterable, List, Union
import warnings

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


class CharTokenizer(AbsTokenizer):
    def __init__(
        self,
        char_list: Union[Path, str, Iterable[str]] = None,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
        self.space_symbol = space_symbol
        if isinstance(char_list, (Path, str)):
            char_list = Path(char_list)
            with char_list.open("r", encoding="utf-8") as f:
                chars = [line.rstrip() for line in f]
        else:
            chars = list(char_list) if char_list is not None else []

        for x in ["<unk>", space_symbol]:
            if x not in chars:
                chars.insert(-2, x)
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.char2idx = {c: i for i, c in self.idx2char.items()}

        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set(x for x in chars if len(x) > 1)
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                self.non_linguistic_symbols = set(line.rstrip() for line in f)
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)

        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols or w == "<unk>":
                        tokens.append(line[:len(w)])
                    line = line[len(w):]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = self.space_symbol
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return "".join(t if t != self.space_symbol else " " for t in tokens)

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        unk_id = self.char2idx["<unk>"]
        return [self.char2idx.get(tok, unk_id) for tok in tokens]

    def ids2tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.idx2char[i] for i in ids]

    def __len__(self):
        return len(self.idx2char)


class SentencepieceTokenizer(AbsTokenizer):
    def __init__(self, model: Union[Path, str]):
        assert check_argument_types()
        self.model = str(model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model)

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
        return [self.sp.PieceToId(tok) for tok in tokens]

    def ids2tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.sp.IdToPiece(i) for i in ids]

    def text2ids(self, line: str) -> List[int]:
        return self.sp.EncodeAsIds(line)

    def ids2text(self, ids: Iterable[int]) -> str:
        return self.sp.DecodeIds(ids)

    def __len__(self):
        return self.sp.get_piece_size()


class WordTokenizer(AbsTokenizer):
    def __init__(
        self,
        vocab: Union[Path, str, Iterable[str]] = None,
        delimiter: str = None,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
        self.delimiter = " " if delimiter is None else delimiter

        if not remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            warnings.warn(
                "non_linguistic_symbols is only used "
                "when remove_non_linguistic_symbols = True"
            )

        if isinstance(vocab, (Path, str)):
            vocab = Path(vocab)
            with vocab.open("r", encoding="utf-8") as f:
                words = [line.rstrip() for line in f]
        else:
            words = list(vocab) if vocab is not None else []
        if "<unk>" not in words:
            words.insert(-2, "<unk>")
        self.idx2word = {i: c for i, c in enumerate(words)}
        self.word2idx = {c: i for i, c in self.idx2word.items()}

        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set(
                x for x in words if x != "<unk>" and re.fullmatch("<.*>", x))
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                self.non_linguistic_symbols = set(line.rstrip() for line in f)
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return f'{self.__class__.__name__}(delimiter="{self.delimiter}")'

    def text2tokens(self, line: str) -> List[str]:
        def remove(t):
            return self.remove_non_linguistic_symbols and t in self.non_linguistic_symbols
        return [t for t in line.split(self.delimiter) if not remove(t)]

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.delimiter.join(tokens)

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        unk_id = self.word2idx["<unk>"]
        return [self.word2idx.get(tok, unk_id) for tok in tokens]

    def ids2tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.idx2word[i] for i in ids]

    def __len__(self):
        return len(self.idx2word)
