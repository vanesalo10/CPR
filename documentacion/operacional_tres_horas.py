#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  operacional_tres_horas.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import pandas as pd
import cprv1.cprv1 as cpr
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.core.groupby import DataError
# 1)logea el tiempo que se demora generar todo
self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
inicia_cronometro = datetime.datetime.now()
end = datetime.datetime.now()
start = end - datetime.timedelta(hours = 3)
#paths
log_path = '/media/nicolas/Home/Jupyter/MarioLoco/logs/tres_horas.txt' #1)
folder_path = '/media/nicolas/Home/Jupyter/MarioLoco/real_time/'
folders = ['tres_horas','diario','tres_dias','treinta_dias']
resolution = ['3h']
for folder,resolution in zip(folders,resolution):
    for codigo in self.infost.index:
        try:
            obj = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
            level = obj.level(start,end).resample('5min',how='mean')
            path = '/media/nicolas/Home/Jupyter/MarioLoco/ultimos_levantamientos'
            levantamiento = pd.read_csv('%s/%s.csv'%(path,codigo),index_col=0)
            filepath = '%s%s/%s.png'%(folder_path,folder,obj.info.slug)
            obj.plot_operacional(level/100.0,levantamiento,resolution,filepath=filepath)
            os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/%s'%(filepath,folder))
        except DataError:
            print 'data error in %s'%codigo
            
termina_cronometro = datetime.datetime.now() 
took = (termina_cronometro-inicia_cronometro)
print(took)
log = pd.Series.from_csv(log_path)
log[datetime.datetime.now()] = took.seconds/60.
log.to_csv(log_path)
print(os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/tres_horas/time_execution_log.txt'%log_path))
