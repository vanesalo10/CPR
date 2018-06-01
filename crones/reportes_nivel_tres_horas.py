#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import cprv1.cprv1 as cpr
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import numpy as np
import multiprocessing
import time
import wmf.wmf as wmf

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
    logging.basicConfig(filename = 'reporte_nivel.log',level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        start = time.time()
        f = orig_func(*args,**kwargs)
        date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        took = time.time()-start
        log = '%s:%s:%.1f sec'%(date,orig_func.__name__,took)
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

        
@logger        
def convert_series_to_risk(self,level):
    '''level: pandas Series, index = codigos de estaciones'''
    risk = level.copy()
    colors = ['green','gold','orange','red','red','black']
    for codigo in level.index:
        try:
            risks = cpr.Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss').risk_levels
            risk[codigo] = colors[int(self.convert_level_to_risk(level[codigo],risks))]
        except:
            risk[codigo] = 'black'
    return risk

@logger
def reporte_lluvia():
    try:
        self = cpr.Nivel(codigo=260,user='sample_user',passwd = 's@mple_p@ss',SimuBasin=True)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(hours=3)
        posterior = end + datetime.timedelta(minutes=10)
        rain = self.radar_rain(start,posterior)
        rain_vect = self.radar_rain_vect(start,posterior)
        codigos = self.infost.index
        df = pd.DataFrame(index = rain_vect.index,columns=codigos)
        for codigo in codigos:
            mask_path = '/media/nicolas/maso/Mario/mask/mask_%s.tif'%(codigo)
            try:
                mask_map = wmf.read_map_raster(mask_path)
                mask_vect = self.Transform_Map2Basin(mask_map[0],mask_map[1])
            except AttributeError:
                print 'mask:%s'%codigo
                mask_vect = None
            if mask_vect is not None:
                mean = []
                for date in rain_vect.index:
                    try:
                        mean.append(np.sum(mask_vect*rain_vect.loc[date])/np.sum(mask_vect))
                    except:
                        print 'mean:%s'%codigo
                if len(mean)>0:
                    df[codigo] = mean
        df_posterior = df.loc[end:]
        plt.rc('font', **{'size'   :16})
        fig = plt.figure(figsize=(20,20))
        fig.subplots_adjust(hspace=1.1)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        suma = (df/1000.).sum().sort_values(ascending=False)
        suma = suma[suma>0.0]
        orden = np.array(suma.index,int)
        suma.index = self.infost.loc[suma.index,'nombre']
        risk = convert_series_to_risk(self,self.level_all(hours=1).iloc[-3:].max())
        dfb = pd.DataFrame(index=suma.index,columns=['rain','color'])
        dfb['rain'] = suma.values
        dfb['color'] = risk.loc[orden].values
        dfb.plot.bar(y='rain', color=[dfb['color']],ax=ax1)
        #suma.plot(kind='bar',color = list(),ax=ax1)
        title = 'start: %s, end: %s'%(start.strftime('%Y-%m-%d %H:%M'),end.strftime('%Y-%m-%d %H:%M'))
        filepath = '/media/nicolas/Home/Jupyter/MarioLoco/reportes/lluvia_en_cuencas.png'
        ax1.set_title(title)
        ax1.set_ylabel('lluvia acumulada\n promedio en la cuenca [mm]')
        suma = (df_posterior/1000.).sum().loc[orden]
        suma.index = self.infost.loc[suma.index,'nombre']
        dfb = pd.DataFrame(index=suma.index,columns=['rain','color'])
        dfb['rain'] = suma.values
        dfb['color'] = risk.loc[orden].values
        dfb.plot.bar(y='rain', color=[dfb['color']],ax=ax2)
        #suma.plot(kind='bar',ax=ax2)
        filepath = '/media/nicolas/Home/Jupyter/MarioLoco/reportes/lluvia_en_cuencas.png'
        ax2.set_title(u'lluvia acumulada en la próxima media hora')
        ax2.set_ylabel('lluvia acumulada\n promedio en la cuenca [mm]')
        ax1.set_ylim(0,30)
        ax2.set_ylim(0,30)
        plt.savefig(filepath,bbox_inches='tight')
        r = os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/reporte_lluvia_cuenca.png'%filepath)
        print 'RESULTADO %s'%r
    except:
        print 'something wrong'
        pass

#PROCESS
self = cpr.Nivel(codigo = 99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
df = data_base_query() #dataframe level
risk_df = convert_to_risk(df.copy()) 
in_risk = risk_report(risk_df) # risk dataframe
# process order according to risk
if __name__ == '__main__':
    p = multiprocessing.Process(target=reporte_lluvia, name="r")
    p.start()
    time.sleep(100) # wait near 5 minutes to kill process
    p.terminate()
    p.join()
    
try:
    df[risk_df.sum(axis=1).sort_values(ascending=False).index] # o
except:
    print 'ordering didnt work'
    pass
#TO KILL PROCESS AFTER CERTAIN PERIOD OF TIME
if __name__ == '__main__':
    p = multiprocessing.Process(target=process_multiple_plots_looping, name="")
    p.start()
    time.sleep(290) # wait near 5 minutes to kill process
    p.terminate()
    p.join()

    
df.to_csv('/media/nicolas/maso/Mario/tres_horas.csv')

#TO KILL PROCESS AFTER CERTAIN PERIOD OF TIME

import cprv1.cprv1 as cpr
import datetime
import os
import matplotlib.pyplot as plt

try:
    self = cpr.Nivel(codigo = 302,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
    end = datetime.datetime.now()
    start = end - datetime.timedelta(hours=2)

    fig = plt.figure(figsize=(20,18))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)


    s = (self.fecha_hora_format_data('ni',start,end)-2095)/100.
    s.abs().plot(fontsize=20,ax=ax1)
    ax1.set_ylabel('Nivel (m)',fontsize=20)
    ax1.set_xlabel('hora',fontsize=20)

    start = end - datetime.timedelta(hours=12)
    s = (self.fecha_hora_format_data('ni',start,end)-2095)/100.
    s.abs().plot(fontsize=20,ax=ax2)
    ax2.set_ylabel('Nivel (m)',fontsize=20)
    ax2.set_xlabel('hora',fontsize=20)
    ax1.set_title('dos horas',fontsize=20)
    ax2.set_title('doce horas',fontsize=20)

    filepath ='/media/nicolas/Home/Jupyter/MarioLoco/reportes/hidroituango.png'
    #plt.suptitle('ver reporte de lluvia en esta misma carpeta')
    plt.savefig(filepath,bbox_inches='tight')
    os.system('scp %s mcano@siata.gov.co:/var/www/mario'%filepath)
except:
    print 'didnt work'
    pass