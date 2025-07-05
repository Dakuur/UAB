# -*- coding: utf-8 -*-

import datetime as dt
from dataclasses import dataclass, field, InitVar
from typing import List, Dict
from datetime import date, datetime, timedelta

@dataclass
class Prestec():
    _codi_usuari: str
    _codi_publicacio: str
    _data_préstec: date
    _data_retorn: date
    _n_exemplar: int = field(default_factory = 0)