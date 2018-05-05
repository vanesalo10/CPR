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
    logging.basicConfig(filename = 'reportes_nivel.log',level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        start = time.time()
        f = orig_func(*args,**kwargs)
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        took = time.time()-start
        log = '%s:%s:%.3f sec'%(date,orig_func.__name__,took)
        print log
        logging.info(log)
        return f
    return wrapper

@logger
def data_base_query():
    return self.level_all()

def convert_to_risk(df):
    df = self.risk_df(df)
    return df[df.columns.dropna()]

@logger
def risk_report(df):
    return self.make_risk_report_current(df)

def plot_level(codigo):
    try:
        resolution='3h' # will be argument later on
        path = '/media/nicolas/Home/Jupyter/MarioLoco'
        if resolution == '3h':
            folder = 'tres_horas'
        obj = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
        levantamiento = pd.read_csv('%s/ultimos_levantamientos/%s.csv'%(path,codigo),index_col=0)
        filepath = '%s/%s.png'%(path+'/real_time/'+folder,obj.info.slug)
        obj.plot_operacional(df[codigo]/100.0,levantamiento,resolution,filepath=filepath)
        r = os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/%s'%(filepath,folder))
    except:
        print 'error in plot %s'%codigo
@logger
def processs_multiple_plots():
    from multiprocessing import Pool
    if __name__ == '__main__':
        p = Pool(10)
        p.map(plot_level, list(df.columns))
        p.close()
        p.join()

@logger
def process_multiple_plots_looping():
    for codigo in df.columns:
        plot_level(codigo)   

#PROCESS
self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
df = data_base_query() #dataframe level
risk_df = convert_to_risk(df.copy()) 
in_risk = risk_report(risk_df) # risk dataframe
try:
    df = df[risk_df.sum(axis=1).sort_values(ascending=False).index] # ordering stations according to risk
except:
    pass
process_multiple_plots_looping()
os.system('scp /home/nicolas/reportes_nivel.log mcano@siata.gov.co:/var/www/mario/realTime')
os.system('scp /home/nicolas/self_code/Crones/niveles_riesgo_error.log mcano@siata.gov.co:/var/www/mario/realTime')
