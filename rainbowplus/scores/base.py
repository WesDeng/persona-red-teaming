
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod


class BaseScore(ABC):
    @abstractmethod
    def get_name(self):
        raise NotImplementedError("get_name() must be implemented in a subclass")

    @abstractmethod
    # Compare two responses and return a score
    def score(self, candidate: str, reference: str):
        raise NotImplementedError("score() must be implemented in a subclass")
