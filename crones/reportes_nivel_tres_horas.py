#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import cprv1.cprv1 as cpr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import numpy as np

def logger(orig_func):
    '''logging decorator, alters function passed as argument and creates
    log file. (contains function time execution)
    Parameters
    ----------
    orig_func : function to pass into decorator
    Returns
    -------
    log file
    '''
    import logging
    from functools import wraps
    import time
    logging.basicConfig(filename = '{}.log'.format(orig_func.__name__),level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        start = time.time()
        f = orig_func(*args,**kwargs)
        took = time.time()-start
        log = 'Ran with kwargs:{},took:{}'.format(kwargs,took)
        print log
        logging.info(log)
        return f
    return wrapper

@logger
def data_base_query(f_name='db',**kwargs):
    return self.level_all()
def convert_to_risk(df):
    df = self.risk_df(df)
    return df[df.columns.dropna()]
@logger
def risk_report(df,f_name = 'report',**kwargs):
    return self.make_risk_report_current(df)
self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
df = data_base_query(**{'f_name':'3h all stations'})
df = convert_to_risk(df)
risk_report(df,**{'f_name':'risk report'})
