#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>

import cprv1.cprv1 as cpr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
#
end = datetime.datetime.now()
start = end - datetime.timedelta(hours=3)
self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
bads = []
filepath = '/media/nicolas/Home/Jupyter/MarioLoco/operacional/tres_horas'
for count,codigo in enumerate(self.infost.index):
    try:
        obj = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
        series = obj.level_local(start,end)
        obj.plot_operacional(series/100.0,
                             window='3h',
                             filepath = '%s/%s.png'%(filepath,obj.codigo))
        plt.close('all')
    except:
        print 'ERROR: codigo: %s'%codigo
        bads.append([codigo,datetime.datetime.now(),start,end])
    timer =  (datetime.datetime.now()-end).seconds/60.0
    format = (codigo,(count+1)*100.0/float(self.infost.index.size),count+1,self.infost.index.size,timer)
    print 'id: %s | %.1f %% | %d out of %d | %.2f minutes'%format
    
try:
    pd.DataFrame(bads,columns = ['codigo','date','start','end']).to_csv('%s/log.csv'%filepath)
except:
    print 'log not saved'
