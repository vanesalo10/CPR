#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>
import matplotlib.dates as mdates
import cprv1.cprv1 as cpr
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import os

def rain_report(self,date):
    date = pd.to_datetime(date)
    start = date-datetime.timedelta(minutes=150)# 3 horas atras
    end = date+datetime.timedelta(minutes=30) 
    minutes = 180
    #rain
    vec = self.radar_rain(start,end,ext='.bin')
    current_vect = vec.drop(vec.loc[date:].index).sum().values/1000
    future_vect = vec.drop(vec.loc[:date].index).sum().values/1000
    folder_path = '/media/nicolas/Home/Jupyter/MarioLoco/reportes_lluvia/%s'%date.strftime('%Y%m%d')
    os.system('mkdir %s'%folder_path)
    filepath = '%s/%s'%(folder_path,self.file_format(start,end))
    self.plot_rain_future(current_vect,future_vect,filepath = filepath+'_rain')
    # level
    mean_rain = self.radar_rain(start,end)*12.0
    #series = self.level_local(start,date)
    series = self.level(start,date).resample('5min').mean()
    series.index.name = ''
    self.plot_level_report(series,mean_rain,self.risk_levels)
    plt.gca().set_xlim(start,end)
    plt.savefig(filepath+'_level.png',bbox_inches='tight')
    rain_report_reportlab(self,filepath,date)
    print('%s-%s minutes'%(self.codigo,(datetime.datetime.now()-date).seconds/60.0))
    return filepath
    
def rain_report_reportlab(self,filepath,date):
    avenir_book_path = '/media/nicolas/Home/Jupyter/MarioLoco/Tools/AvenirLTStd-Book.ttf'
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate,Paragraph, Table, TableStyle
    from IPython.display import IFrame
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.pdfgen import canvas
    pdfmetrics.registerFont(TTFont('AvenirBook', avenir_book_path))
    current_vect_title = 'Lluvia acumulada en la cuenca en las últimas dos horas'
    # REPORLAB
    pdf = canvas.Canvas(filepath+'_report.pdf',pagesize=(900,1200))
    cx = 0
    cy = 900
    pdf.drawImage(filepath+'_rain.png',60,650,width=830,height=278)
    pdf.drawImage(filepath+'_level.png',20,270+20,width=860,height=280)
    pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/pie.png',0,0,width=905,height=145.451)
    pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/cabeza.png',0,1020,width=905,height=180)
    pdf.setFillColor('#%02x%02x%02x' % (8,31,45))
    pdf.setFont("AvenirBook", 23)
    pdf.drawString(240,1045,u'Estación %s - %s'%(self.info.nombre,date.strftime('%d %B de %Y')))
    # últimas dos horas
    pdf.setFillColor('#%02x%02x%02x' % (8,31,45))
    pdf.setFont("AvenirBook", 20)
    pdf.drawString(90,520+300+30+100+30,'Lluvia acumulada en la cuenca')
    pdf.drawString(90,520+300+10+100+20,'en las últimas dos horas')
    # Futuro
    distance = 430
    pdf.drawString(90+distance-10,520+300+100+40+20,'Lluvia acumulada en la cuenca')
    pdf.drawString(90+distance-10,520+300+10+100+20,'en la próxima media hora')
    pdf.drawString(90,270+280+10+20,'Profundidad de la lámina de agua e intensidad promedio en la cuenca')

    #pdf.setFont("AvenirBook", 15)

    pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/leyenda.png',67,180,width=800,height=80)
    #N1
    pdf.setFillColor('#%02x%02x%02x' % (8,31,45))
    pdf.setFont("AvenirBook", 15)
    pdf.drawString(80,210,'Nivel seguro')
    #pdf.drawString(90+distance,520+300+10+100,'en la próxima media hora')
    factor = 143
    pdf.drawString(80+factor,210,'Nivel de ')
    pdf.drawString(80+factor,190,'alerta')
    #N2
    factor = 287
    pdf.drawString(80+factor,210,'Inundación')
    pdf.drawString(80+factor,190,'menor')
    #N4
    factor = 430
    pdf.drawString(80+factor,210,'Inundación')
    pdf.drawString(80+factor,190,'mayor')
    #N4
    factor = 430
    y = 30
    pdf.drawString(700,210+y,'Intensidad de lluvia')
    pdf.drawString(700,180+y,'Profundidad')
    pdf.drawString(700,160+y-5,'Profundidad actual')

    #pdf.drawString(90+distance,520+300+10+100,'en la próxima media hora')

    pdf.showPage()
    pdf.save()
    #os.system('scp %s mcano@siata.gov.co:/var/www/mario/rainReport/%d'%(ruteSave[:-3]+'pdf',self.codigo))
    # LOG
    folderpath = pd.to_datetime(date).strftime('%Y%m%d')
    r = os.system('scp %s mcano@siata.gov.co:/var/www/mario/reportes_lluvia/%s'%(filepath+'_report.pdf',folderpath))
    if r == 0:
	'no copia el link'

def level_all(self,start=None,end = None,hours=3):
    if start:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
    else:
        end = pd.to_datetime(self.round_time())
        start = end - datetime.timedelta(hours = hours)
    codigos = self.infost.index
    df = pd.DataFrame(index = pd.date_range(start,end,freq='5min'),columns = codigos)
    for codigo in codigos:
        try:
            level = cpr.Nivel(codigo=codigo,user='sample_user',passwd='s@mple_p@ss').level(start,end).resample('5min').mean()
            df[codigo] = level
        except:
            pass
    return df


self = cpr.Nivel(codigo=99,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
end = self.round_time(datetime.datetime.now())
start = self.round_time(end - datetime.timedelta(days=30))
df = level_all(self,start,end)
df = df.resample('5min').mean()
dfr = self.risk_df(df)
dfr[dfr<2]=np.NaN

for codigo in dfr.index:
    try:
        obj = cpr.Nivel(codigo=codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=True)
        fechas = dfr.loc[codigo].dropna().index
        for fecha in fechas:
            try:
                rain_report(obj,fecha)
                print 'FINALIZADO: %s - %s'%(codigo,fecha)
            except:
		print 'no funciona %s'%fecha
    except:
        print 'NO FUNCIONA: %s'%codigo
