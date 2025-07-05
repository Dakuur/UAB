# -*- coding: utf-8 -*-

from publicacio import Publicacio
import datetime as dt
from dataclasses import dataclass, field, InitVar
from typing import List, Dict
from datetime import date, datetime, timedelta

@dataclass
class Revista(Publicacio):
    _periodicitat: str
    _exemplars: List[int]