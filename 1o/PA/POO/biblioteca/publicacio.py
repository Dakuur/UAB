# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from datetime import date, datetime, timedelta

@dataclass
class Publicacio():
    _codi: str
    _titol: str