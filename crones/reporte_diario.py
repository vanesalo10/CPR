#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>

import cprv1.cprv1 as cpr
import datetime
import pandas as pd
self = cpr.Nivel(codigo=99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
date = datetime.datetime.now()
self.reporte_diario(date)
#for fecha in pd.date_range(date-datetime.timedelta(days=2),date,freq='D'):
#    print fecha
#    self.reporte_diario(fecha)
