# -*- coding: utf-8 -*-

from publicacio import Publicacio
import datetime as dt
from dataclasses import dataclass, field, InitVar
from datetime import date, datetime, timedelta

@dataclass
class Llibre(Publicacio):
    _autor: str
    _n_copies: int
    _n_dies: int