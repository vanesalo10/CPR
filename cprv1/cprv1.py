#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  CRP.py
#
#  Copyright 2018 MCANO <mario.cano@siata.gov.co>

import MySQLdb
import pandas as pd
import numpy as np
import datetime
import math
import time
import mysql.connector
from sqlalchemy import create_engine
import os
import warnings
import static as st
import bookplots as bp
import information as info
from wmf import wmf
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
import locale
import matplotlib.dates as mdates
#reportlab libraries
#from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph, Table, TableStyle
from IPython.display import IFrame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#PYTHON CONFIGURATION
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
# configure locale for data in spanish
#sudo locale-gen es_ES.UTF-8
#sudo dpkg-reconfigure locales
# Siata settings
#plt.rc('font', family=fm.FontProperties(fname='/media/nicolas/maso/Mario/tools/AvenirLTStd-Book.ttf',).get_name())
typColor = '#%02x%02x%02x' % (8,31,45)
plt.rc('axes',labelcolor=typColor)
plt.rc('axes',edgecolor=typColor)
plt.rc('text',color= typColor)
plt.rc('xtick',color=typColor)
plt.rc('ytick',color=typColor)
#font = {'size'   : 14}
#plt.rc('font', **font)

class SqlDb:
    '''
    Class para manipular las bases de datos SQL
    '''
    str_date_format = '%Y-%m-%d %H:%M:00'

    def __init__(self,dbname,user,host,passwd,port,table=None,codigo=None,
                *keys,**kwargs):
        '''
        instance and properties
        '''
        self.table  = table
        self.host   = host
        self.user   = user
        self.passwd = passwd
        self.dbname = dbname
        self.port   = port
        self.codigo = codigo

    def __repr__(self):
        '''string to recreate the object'''
        return "codigo = {}".format(self.codigo)

    def __str__(self):
        '''string to recreate the main information of the object'''
        return 'dbname: {}, user: {}'.format(self.dbname,self.user)

    @property
    def conn_db(self):
        '''
        Engine connection: makes possible connection with SQL database
        '''
        conn_db = MySQLdb.connect(self.host,self.user,self.passwd,self.dbname,charset='utf8')
        return conn_db

    def logger(self,function,status,message):
        '''
        Logs methods performance
        Returns
        -------
        string comma separated values'''
        now = datetime.datetime.now().strftime(self.str_date_format)
        return '%s,%s,%s,%s'%(now,self.user,function,status,message)

    def read_sql(self,sql,close_db=True,*keys,**kwargs):
        '''
        Read SQL query or database table into a DataFrame.
        Parameters
        ----------
        sql : string SQL query or SQLAlchemy Selectable (select or text object)
            to be executed, or database table name.

        keys and kwargs = ( sql, con, index_col=None, coerce_float=True,
                            params=None, parse_dates=None,columns=None,
                            chunksize=None)
        Returns
        -------
        DataFrame
        '''
        conn_db = MySQLdb.connect(self.host,self.user,self.passwd,self.dbname)
        df = pd.read_sql(sql,conn_db,*keys,**kwargs)
        if close_db == True:
            conn_db.close()
        return df

    def execute_sql(self,query,close_db=True):
        '''
        Execute SQL query or database table into a DataFrame.
        Parameters
        ----------
        query : string SQL query or SQLAlchemy Selectable (select or text object)
            to be executed, or database table name.
        keys = (sql, con, index_col=None, coerce_float=True, params=None,
        parse_dates=None,
        columns=None, chunksize=None)
        Returns
        -------
        DataFrame'''
        conn_db = self.conn_db
        conn_db.cursor().execute(query)
        conn_db.commit()
        if close_db == True:
            conn_db.close ()
        #print (self.logger('execute_mysql','execution faile','worked',query))

    def insert_data(self,fields,values,*keys,**kwargs):
        '''
        inserts data into SQL table from list of fields and values
        Parameters
        ----------
        fields   = list of fields names from SQL db
        values   = list of values to be inserted
        Example
        -------
        insert_data(['fecha','nivel'],['2017-07-13',0.5])
        '''
        values = str(values).strip('[]')
        fields = str(fields).strip('[]').replace("'","")
        execution = 'INSERT INTO %s (%s) VALUES (%s)'%(self.table,fields,values)
        self.execute_sql(execution,*keys,**kwargs)

    def update_data(self,field,value,pk,*keys,**kwargs):
        '''
        Update data into SQL table
        Parameters
        ----------
        fields   = list of fields names from SQL db
        values   = list of values to be inserted
        pk       = primary key from table
        Example
        -------
        update_data(['nivel','prm'],[0.5,0.2],1025)
        '''
        query = "UPDATE %s SET %s = '%s' WHERE id = '%s'"%(self.table,field,value,pk)
        self.execute_sql(query,*keys,**kwargs)

    def read_boundary_date(self,how,date_field_name = 'fecha'):
        '''
        Gets boundary date from SQL table based on DateField or DatetimeField name
        Parameters
        ----------
        how             = method to get boundary, could be max or min
        date_field_name = field name in Table
        Example
        -------
        read_bound_date('min')
        '''
        format = (how,date_field_name,self.table,name,codigo)
        return self.read_sql("select %s(%s) from %s where codigo='%s'"%format).loc[0,'%s(fecha)'%how]

    def df_to_sql(self,df,chunksize=20000,*keys,**kwargs):
        '''Replaces existing table with dataframe
        Parameters
        ----------
        df        = Pandas DataFrame to replace table
        chunksize = If not None, then rows will be written in batches
        of this size at a time
        '''
        format = (self.user,self.passwd,self.host,self.port,)
        engine = create_engine('mysql+mysqlconnector://%s:%s@%s:%s/cpr'%format,echo=False)
        df.to_sql(name      = self.table,
                  con       = engine,
                  if_exists = 'replace',
                  chunksize = chunksize,
                  index     = False,
                  *keys,**kwargs)

    def bound_date(self,how,date_field_name='fecha'):
        '''
        Gets firs and last dates from date field name of SQL table
        Parameters
        ----------
        how                = min or max (ChoiseField),
        date_field_name    = field name of SQL table, containing datetime,
        timestamp or other time formats
        Returns
        ----------
        DateTime object
        '''
        format = (how,date_field_name,self.table,self.codigo)
        return self.read_sql("select %s(%s) from %s where codigo='%s'"%format).loc[0,'%s(%s)'%(how,date_field_name)]

    @property
    def info(self):
        '''
        Gets full information from single station
        ---------
        pd.Series
        '''
        query = "SELECT * FROM %s"%self.table
        return self.read_sql(query).T[0]

    def update_series(self,series,field):
        '''
        Update table from pandas time Series
        Parameters
        ----------
        series   = pandas time series with datetime or timestamp index
        and frequency = '5min'
        field    = field to be update
        Example
        value = series[fecha]
        ----------
        series = pd.Series(...,index=pd.date_range(...))
        update_series(series,'nivel')
        this updates the field nivel
        '''
        pk = self.id_df
        t  = datetime.datetime.now()
        for count,fecha in enumerate(series.index):
            value = series[fecha]
            if math.isnan(value):
                pass
            else:
                id    = pk[fecha]
                self.update_data(field,value,id)
            timer =  (datetime.datetime.now()-t).seconds/60.0
            format = (self.codigo,(count+1)*100.0/float(series.index.size),count+1,series.index.size,timer)
            print 'id: %s | %.1f %% | %d out of %d | %.2f minutes'%format

    @staticmethod
    def fecha_hora_query(start,end):
        '''
        Efficient way to query in tables with fields fecha,hora
        such as table datos
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        Alternative query between two datetime objects
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        def f(date):
            return tuple([date.strftime('%Y-%m-%d')]*2+[date.strftime('%H:%M:00')])
        query = "("+\
                "((fecha>'%s') or (fecha='%s' and hora>='%s'))"%f(start)+" and "+\
                "((fecha<'%s') or (fecha='%s' and hora<='%s'))"%f(end)+\
                ")"
        return query

    def fecha_hora_format_data(self,field,start,end,**kwargs):
        '''
        Gets pandas Series with data from tables with
        date format fecha and hora detached, and filter
        bad data
        Parameters
        ----------
        field        : Sql table field name
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas time Series
        '''
        start= pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:00')
        end = pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:00')
        format = (field,self.codigo,self.fecha_hora_query(start,end))
        sql = SqlDb(codigo = self.codigo,**info.REMOTE)
        if kwargs.get('calidad'):
            df = sql.read_sql("SELECT fecha,hora,%s from datos WHERE calidad = '1' and cliente = '%s' and %s"%format)
        else:
            df = sql.read_sql("SELECT fecha,hora,%s from datos WHERE cliente = '%s' and %s"%format)
        # converts centiseconds in 0
        try:
            df['hora'] = df['hora'].apply(lambda x:x[:-3]+':00')
        except TypeError:
            df['hora']=df['hora'].apply(lambda x:str(x)[-8:-8+5]+':00')
            df['fecha'] = df['fecha'].apply(lambda x:x.strftime('%Y-%m-%d'))
        # concatenate fecha and hora fields, and makes nan bad datetime indexes
        df.index= pd.to_datetime(df['fecha'] + ' '+ df['hora'],errors='coerce')
        df = df.sort_index()
        # removes nan
        df = df.loc[df.index.dropna()]
        # masks duplicated index
        df[df.index.duplicated(keep=False)]=np.NaN
        df = df.dropna()
        # drops coluns fecha and hora
        df = df.drop(['fecha','hora'],axis=1)
        # reindex to have all indexes in full time series
        new_index = pd.date_range(start,end,freq='min')
        series = df.reindex(new_index)[field]
        return series

    def round_time(self,date = datetime.datetime.now(),round_mins=5):
        mins = date.minute - (date.minute % round_mins)
        return datetime.datetime(date.year, date.month, date.day, date.hour, mins) + datetime.timedelta(minutes=round_mins)



class Nivel(SqlDb,wmf.SimuBasin):
    '''
    Provide functions to manipulate data related
    to a level sensor and its basin.
    '''
    local_table  = 'estaciones_estaciones'
    remote_table = 'estaciones'
    def __init__(self,user,passwd,codigo = None,SimuBasin = False,remote_server = info.REMOTE,**kwargs):
        '''
        The instance inherits modules to manipulate SQL
        data and uses (hidrology modeling framework) wmf
        Parameters
        ----------
        codigo        : primary key
        remote_server :
        local_server  : database kwargs to pass into the Sqldb class
        nc_path       : path of the .nc file to set wmf class
        '''
        self.remote_server = remote_server
        self.data_path ='/media/nicolas/maso/Mario/'
        self.rain_path = self.data_path + 'user_output/radar/'
        self.radar_path = '/media/nicolas/Home/nicolas/101_RadarClass/'
        if not kwargs:
            kwargs = info.LOCAL
        SqlDb.__init__(self,codigo=codigo,user=user,passwd=passwd,**kwargs)
        if SimuBasin:
            query = "SELECT nc_path FROM %s WHERE codigo = '%s'"%(self.local_table,self.codigo)
            try:
                nc_path = self.read_sql(query)['nc_path'][0]
            except:
                nc_path = self.data_path + 'basins/%s.nc'%self.codigo
            wmf.SimuBasin.__init__(self,rute=nc_path)

    	self.colores_siata = [[0.69,0.87,0.93],[0.61,0.82,0.88],[0.32,0.71,0.77],[0.21,0.60,0.65],\
                          [0.0156,0.486,0.556],[0.007,0.32,0.36],[0.0078,0.227,0.26]]

    @property
    def info(self):
        query = "SELECT * FROM %s WHERE clase = 'Nivel' and codigo='%s'"%(self.local_table,self.codigo)
        s = self.read_sql(query).T
        return s[s.columns[0]]

    @property
    def infost(self):
        '''
        Gets full information from all stations
        Returns
        ---------
        pd.DataFrame
        '''
        query = "SELECT * FROM %s WHERE clase ='Nivel'"%(self.local_table)
        return self.read_sql(query).set_index('codigo')

    @staticmethod
    def get_radar_rain(start,end,nc_path,radar_path,save,
                    converter = 'RadarConvStra2Basin2.py',
                    utc=False,
                    dt = 300,*keys,**kwargs):
        '''
        Convert radar rain to basin
        Parameters
        ----------
        start         : inicial date
        end           : final date
        nc_path       : path to nc basin file
        radar_path    : path to radar data
        save          : path to save
        converter     : path of main rain converter script,
                        default RadarConvStra2Basin2.py
        utc           : if radar data is in utc
        dt            : timedelta, default = 5 minutes
        Returns
        ----------
        bin, hdr files with rain data
        '''
        start = pd.to_datetime(start); end = pd.to_datetime(end)
        if utc ==True:
            delay = datetime.timedelta(hours=5)
            start = start+delay
            end = end + delay
        hora_inicial = start.strftime('%H:%M')
        hora_final = end.strftime('%H:%M')
        format = (
                converter,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
                nc_path,
                radar_path,
                save,
                dt,
                hora_inicial,
                hora_final
                 )
        query = '%s %s %s %s %s %s -t %s -v -s -1 %s -2 %s'%format
        output = os.system(query)
        print query
	if output != 0:
            print 'ERROR: something went wrong'
        return query

    @staticmethod
    def hdr_to_series(path):
        '''
        Reads hdr rain files and converts it into pandas Series
        Parameters
        ----------
        path         : path to .hdr file
        Returns
        ----------
        pandas time Series with mean radar rain
        '''
        s =  pd.read_csv(path,skiprows=5,usecols=[2,3]).set_index(' Fecha ')[' Lluvia']
        s.index = pd.to_datetime(map(lambda x:x.strip()[:10]+' '+x.strip()[11:],s.index))
        return s

    @staticmethod
    def hdr_to_df(path):
        '''
        Reads hdr rain files and converts it into pandas DataFrame
        Parameters
        ----------
        path         : path to .hdr file
        Returns
        ----------
        pandas DataFrame with mean radar rain
        '''
        if path.endswith('.hdr') != True:
            path = path+'.hdr'
        df = pd.read_csv(path,skiprows=5).set_index(' Fecha ')
        df.index = pd.to_datetime(map(lambda x:x.strip()[:10]+' '+x.strip()[11:],df.index))
        df = df.drop('IDfecha',axis=1)
        df.columns = ['record','mean_rain']
        return df

    def bin_to_df(self,path,start=None,end=None,**kwargs):
        '''
        Reads rain fields (.bin) and converts it into pandas DataFrame
        Parameters
        ----------
        path         : path to .hdr and .bin file
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame with mean radar rain
        Note
        ----------
        path without extension, ejm folder_path/file not folder_path/file.bin,
        if start and end is None, the program process all the data
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        df = self.hdr_to_df(path)
        if (start is not None) and (end is not None):
            df = df.loc[start:end]
        df = df[df['record']!=1]
        records = df['record'].values
        rain_field = []
        for count,record in enumerate(records):
            rain_field.append(wmf.models.read_int_basin('%s.bin'%path,record,self.ncells)[0])
            count = count+1
            format = (count*100.0/len(records),count,len(records))
            #print("progress: %.1f %% - %s out of %s"%format)
        return pd.DataFrame(np.matrix(rain_field),index=df.index)

    def file_format(self,start,end):
        '''
        Returns the file format customized for siata for elements containing
        starting and ending point
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        file format with datetimes like %Y%m%d%H%M
        Example
        ----------
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        format = '%Y%m%d%H%M'
        return '%s-%s-%s-%s'%(start.strftime(format),end.strftime(format),self.codigo,self.user)

    def file_format_date_to_datetime(self,string):
        '''
        Transforms string in file_format like string to datetime object
        Parameters
        ----------
        string         : string object in file_format like time object
        Returns
        ----------
        datetime object
        Example
        ----------
        In : self.file_format_date_to_datetime('201707141212')
        Out: Timestamp('2017-07-14 12:12:00')
        '''
        format = (string[:4],string[4:6],string[6:8],string[8:10],string[10:12])
        return pd.to_datetime("%s-%s-%s %s:%s"%format)

    def file_format_to_variables(self,string):
        '''
        Splits file name string in user and datetime objects
        Parameters
        ----------
        string         : file name
        Returns
        ----------
        (user,start,end) - (string,datetime object,datetime object)
        '''
        string = string[:string.find('.')]
        start,end,codigo,user = list(x.strip() for x in string.split('-'))
        start,end = self.file_format_date_to_datetime(start),self.file_format_date_to_datetime(end)
        return start,end,int(codigo),user

    def check_rain_files(self,start,end):
        '''
        Finds out if rain data has already been processed
        start        : initial date
        end          : final date
        Returns
        ----------
        file path or None for no coincidences
        '''
        def todate(date):
            return pd.to_datetime(pd.to_datetime(date).strftime('%Y-%m-%d %H:%M'))
        start,end = todate(start),todate(end)
        files = os.listdir(self.rain_path)
        if files:
            #print files
            for file in files:
                comienza,finaliza,codigo,usuario = self.file_format_to_variables(file)
                if (comienza<=start) and (finaliza>=end) and (codigo==self.codigo):
                    file =  file[:file.find('.')]
                    print file
                    break
                else:
                    file = None
        else:
            file = None
        return file

    def radar_rain(self,start,end,ext='.hdr',nc_path='default'):
        '''
        Reads rain fields (.bin or .hdr)
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame or Series with mean radar rain
        '''
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        file = self.check_rain_files(start,end)
        if file:
            file = self.rain_path+file
            if ext == '.hdr':
                obj =  self.hdr_to_series(file+'.hdr')
            else:
                print file
                obj =  self.bin_to_df(file)
            obj = obj.loc[start:end]
        else:
            print 'converting rain data, it may take a while'
            converter = '/media/nicolas/Home/Jupyter/MarioLoco/repositories/CPR/cprv1/RadarConvStra2Basin2.py'
            #converter = '/home/nicolas/self_code/RadarConvStra2Basin3.py'
            save =  '%s%s'%(self.rain_path,self.file_format(start,end))
            if nc_path == 'default':
                nc_path = self.info.nc_path
            self.get_radar_rain(start,end,nc_path,self.radar_path,save,converter=converter,utc=True)
            print file
            file = self.rain_path + self.check_rain_files(start,end)
            if ext == '.hdr':
                obj =  self.hdr_to_series(file+'.hdr')
            else:
                obj =  self.bin_to_df(file)
            obj = obj.loc[start:end]
        return obj

    def radar_rain_vect(self,start,end,**kwargs):
        '''
        Reads rain fields (.bin)
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame with datetime index and basin radar fields
        '''
        return self.radar_rain(start,end,ext='.bin',**kwargs)

    def sensor(self,start,end,**kwargs):
        '''
        Reads remote sensor level data
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas time series
        '''
        sql = SqlDb(codigo = self.codigo,**self.remote_server)
        s = sql.fecha_hora_format_data(['pr','NI'][self.info.tipo_sensor],start,end,**kwargs)
        return s

    def level(self,start,end,offset='new',**kwargs):
        '''
        Reads remote level data
        Parameters
        ----------
        start        : initial date
        end          : final date
        Returns
        ----------
        pandas DataFrame with datetime index and basin radar fields
        '''
        s = self.sensor(start,end,**kwargs)
        if offset == 'new':
            serie = self.info.offset - s
            serie[serie>=self.info.offset_old] = np.NaN
        else:
            serie = self.info.offset_old - s
            serie[serie>=self.info.offset_old] = np.NaN
        serie[serie<=0.0] = np.NaN
        return serie

    def offset_remote(self):
        remote = SqlDb(**self.remote_server)
        query = "SELECT codigo,fecha_hora,offset FROM historico_bancallena_offset"
        df = remote.read_sql(query).set_index('codigo')
        try:
            offset = float(df.loc[self.codigo,'offset'])
        except TypeError:
            offset =  df.loc[self.codigo,['fecha_hora','offset']].set_index('fecha_hora').sort_index()['offset'][-1]
        return offset


    def mysql_query(self,query,toPandas=True):
        conn_db = MySQLdb.connect(self.host, self.user, self.passwd, self.dbname)
        db_cursor = conn_db.cursor ()
        db_cursor.execute (query)
        if toPandas == True:
            data = pd.DataFrame(np.matrix(db_cursor.fetchall()))
        else:
            data = db_cursor.fetchall()
        conn_db.close()
        return data

    def last_bat(self,x_sensor):
	obj = Nivel(**info.REMOTE)
        dfl = obj.mysql_query('select * from levantamiento_aforo_nueva')
        dfl.columns = obj.mysql_query('describe levantamiento_aforo_nueva')[0].values
        dfl = dfl.set_index('id_aforo')
        for id_aforo in list(set(dfl.index)):
            id_estacion_asociada,fecha = obj.mysql_query("SELECT id_estacion_asociada,fecha from aforo_nueva where id_aforo = %s"%id_aforo,toPandas=False)[0]
            dfl.loc[id_aforo,'id_estacion_asociada'] = int(id_estacion_asociada)
            dfl.loc[id_aforo,'fecha'] = fecha
        dfl = dfl.reset_index().set_index('id_estacion_asociada')
        lev = dfl[dfl['fecha']==max(list(set(pd.to_datetime(dfl.loc[self.codigo,'fecha'].values))))][['x','y']].astype('float')
        cond = (lev['x']<x_sensor).values
        flag = cond[0]
        for i,j in enumerate(cond):
            if j==flag:
                pass
            else:
                point = (tuple(lev.iloc[i-1].values),tuple(lev.iloc[i].values))
            flag = j
        intersection = self.line_intersection(point,((x_sensor,0.1*lev['y'].min()),(x_sensor,1.1*lev['y'].max(),(x_sensor,))))
        lev = lev.append(pd.DataFrame(np.matrix(intersection),index=['x_sensor'],columns=['x','y'])).sort_values('x')
        lev['y'] = lev['y']-intersection[1]
        lev.index = range(1,lev.index.size+1)
        return lev

    def get_sections(self,levantamiento,level):
            hline = ((levantamiento['x'].min()*1.1,level),(levantamiento['x'].max()*1.1,level)) # horizontal line
            lev = pd.DataFrame.copy(levantamiento) #df to modify
            #PROBLEMAS EN LOS BORDES
            borderWarning = 'Warning:\nProblemas de borde en el levantamiento'
            if lev.iloc[0]['y']<level:
                print '%s en banca izquierda'%borderWarning
                lev = pd.DataFrame(np.matrix([lev.iloc[0]['x'],level]),columns=['x','y']).append(lev)
            if lev.iloc[-1]['y']<level:
                print '%s en banca derecha'%borderWarning
                lev = lev.append(pd.DataFrame(np.matrix([lev.iloc[-1]['x'],level]),columns=['x','y']))
            condition = (lev['y']>=level).values
            flag = condition[0]
            nlev = []
            intCount = 0
            ids=[]
            for i,j in enumerate(condition):
                if j==flag:
                    ids.append(i)
                    nlev.append(list(lev.iloc[i].values))
                else:
                    intCount+=1
                    ids.append('Point %s'%intCount)
                    line = (list(lev.iloc[i-1].values),list(lev.iloc[i].values)) #  #puntoA
                    inter = self.line_intersection(line,hline)
                    nlev.append(inter)
                    ids.append(i)
                    nlev.append(list(lev.iloc[i].values))
                flag = j
            df = pd.DataFrame(np.matrix(nlev),columns=['x','y'],index=ids)
            dfs = []
            for i in np.arange(1,100,2)[:intCount/2]:
                dfs.append(df.loc['Point %s'%i:'Point %s'%(i+1)])
            return dfs

    @staticmethod
    def get_area(x,y):
        '''Calcula las áreas y los caudales de cada
        una de las verticales, con el método de mid-section
        Input:
        x = Distancia desde la banca izquierda, type = numpy array
        y = Produndidad
        Output:
        area = Área de la subsección
        Q = Caudal de la subsección
        '''
        # cálculo de áreas
        d = np.absolute(np.diff(x))/2.
        b = x[:-1]+ d
        area = np.diff(b)*y[1:-1]
        area = np.insert(area, 0, d[0]*y[0])
        area = np.append(area,d[-1]*y[-1])
        area = np.absolute(area)
        # cálculo de caudal
        return area

    def get_areas(self,dfs):
        area = 0
        for df in dfs:
            area+=sum(self.get_area(df['x'].values,df['y'].values))
        return area

    @staticmethod
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
           raise Exception('lines do not intersect')
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return (x, y)

    def longitude_latitude_basin(self):
        mcols,mrows = wmf.cu.basin_2map_find(self.structure,self.ncells)
        mapa,mxll,myll=wmf.cu.basin_2map(self.structure,self.structure[0],mcols,mrows,self.ncells)
        longs = np.array([mxll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(mcols)])
        lats  = np.array([myll+0.5*wmf.cu.dx+i*wmf.cu.dx for i in range(mrows)])
        return longs,lats

    def basin_mappable(self,vec=None, extra_long=0,extra_lat=0,perimeter_keys={},contour_keys={},**kwargs):
        longs,lats=self.longitude_latitude_basin()
        x,y=np.meshgrid(longs,lats)
        y=y[::-1]
        # map settings
        m = Basemap(projection='merc',llcrnrlat=lats.min()-extra_lat, urcrnrlat=lats.max()+extra_lat,
            llcrnrlon=longs.min()-extra_long, urcrnrlon=longs.max()+extra_long, resolution='c',**kwargs)
        # perimeter plot
        xp,yp = m(self.Polygon[0], self.Polygon[1])
        m.plot(xp, yp,**perimeter_keys)
        # vector plot
        if vec is not None:
            map_vec,mxll,myll=wmf.cu.basin_2map(self.structure,vec,len(longs),len(lats),self.ncells)
            map_vec[map_vec==wmf.cu.nodata]=np.nan
            xm,ym=m(x,y)
            contour = m.contourf(xm, ym, map_vec.T, 25,**contour_keys)
        else:
            contour = None
        return m,contour

    def adjust_basin(self,rel=0.766,fac=0.0):
        longs,lats = self.longitude_latitude_basin()
        x = longs[-1]-longs[0]
        y = lats[-1] - lats[0]
        if x>y:
            extra_long = 0
            extra_lat = (rel*x-y)/2.0
        else:
            extra_lat=0
            extra_long = (y/(2.0*rel))-(x/2.0)
        return extra_lat+fac,extra_long+fac


    def radar_cmap(self):
        bar_colors=[(255, 255, 255),(0, 255, 255), (0, 0, 255),(70, 220, 45),(44, 141, 29),\
                       (255,255,75),(255,142,0),(255,0,0),(128,0,128),(102,0,102),(255, 153, 255)]
        lev = np.array([0.,1.,5.,10.,20.,30.,45.,60., 80., 100., 150.])
        scale_factor =  ((255-0.)/(lev.max() - lev.min()))
        new_Limits = list(np.array(np.round((lev-lev.min())*\
                                    scale_factor/255.,3),dtype = float))
        Custom_Color = map(lambda x: tuple(ti/255. for ti in x) , bar_colors)
        nueva_tupla = [((new_Limits[i]),Custom_Color[i],) for i in range(len(Custom_Color))]
        cmap_radar =colors.LinearSegmentedColormap.from_list('RADAR',nueva_tupla)
        levels_nuevos = np.linspace(np.min(lev),np.max(lev),255)
        norm_new_radar = colors.BoundaryNorm(boundaries=levels_nuevos, ncolors=256)
        return cmap_radar,levels_nuevos,norm_new_radar

    def level_local(self,start,end,offset='new'):
        if offset=='new':
            offset = self.info.offset
        else:
            offset = self.info.offset_old
        format = (self.codigo,start,end)
        query = "select fecha,nivel from hydro where codigo='%s' and fecha between '%s' and '%s';"%format
        level =  (offset - self.read_sql(query).set_index('fecha')['nivel'])
        level[level>self.risk_levels[-1]*1.2] = np.NaN
        level[level>offset] = np.NaN
        return level

    def convert_level_to_risk(self,value,risk_levels):
        ''' Convierte lamina de agua o profundidad a nivel de riesgo
        Parameters
        ----------
        value : float. Valor de profundidad o lamina de agua
        riskLevels: list,tuple. Niveles de riesgo

        Returns
        -------
        riskLevel : float. Nivel de riesgo
        '''
        if math.isnan(value):
            return np.NaN
        else:
            dif = value - np.array([0]+list(risk_levels))
            return int(np.argmin(dif[dif >= 0]))

    @property
    def risk_levels(self):
        query = "select n1,n2,n3,n4 from estaciones_estaciones where codigo = '%s'"%self.codigo
        return tuple(self.read_sql(query).values[0])

    def risk_level_series(self,start,end):
        return self.level_local(start,end).apply(lambda x: self.convert_level_to_risk(x,self.risk_levels))

    def risk_level_df(self,start,end):
        print 'Making risk dataframe'
        df = pd.DataFrame(index=pd.date_range(start,end,freq='D'),columns=self.infost.index)
        for count,codigo in enumerate(df.columns):
            print "%s | '%.2f %%' - %s out of %s "%(codigo,(count+1)*100.0/df.columns.size,count+1,df.columns.size)
            try:
                clase = Nivel(user=self.user,codigo=codigo,passwd=self.passwd,**info.LOCAL)
                df[codigo] = clase.risk_level_series(start,end).resample('D',how='max')
            except:
                df[codigo] = np.NaN
                print "WARNING: station %s empty,row filled with NaN"%codigo
        print 'risk dataframe finished'
        return df

    def plot_basin_rain(self,vec,cbar=None,ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10,16))
            ax = fig.add_subplot()
        cmap_radar,levels,norm = self.radar_cmap()
        extra_lat,extra_long = self.adjust_basin(fac=0.01)
        mapa,contour = self.basin_mappable(vec,
                                      ax=ax,
                                      extra_long=extra_long,
                                      extra_lat = extra_lat,
                                      contour_keys={'cmap'  :cmap_radar,
                                                    'levels':levels,
                                                    'norm'  :norm},
                                     perimeter_keys={'color':'k'})
        if cbar:
            cbar = mapa.colorbar(contour,location='right',pad="15%")
            cbar.ax.set_title('mm',fontsize=14)
        else:
            cbar = mapa.colorbar(contour,location='right',pad="15%")
            cbar.remove()
            plt.draw()
        mapa.readshapefile(self.info.net_path,'net_path')
        mapa.readshapefile(self.info.stream_path,'stream_path',linewidth=1)
        return mapa

    def plot_section(self,df,*args,**kwargs):
        '''Grafica de la seccion transversal de estaciones de nivel
        |  ----------Parametros
        |  df : dataFrame con el levantamiento topo-batimetrico, columns=['x','y']
        |  level : Nivel del agua
        |  riskLevels : Niveles de alerta
        |  *args : argumentos plt.plot()
        |  **kwargs : xSensor,offset,riskLevels,xLabel,yLabel,ax,groundColor,fontsize,figsize,
        |  Nota: todas las unidades en metros'''
        # Kwargs
        level = kwargs.get('level',None)
        xLabel = kwargs.get('xLabel','Distancia desde la margen izquierda [m]')
        yLabel = kwargs.get('yLabel','Profundidad [m]')
        waterColor = kwargs.get('waterColor',self.colores_siata[1])
        groundColor = kwargs.get('groundColor','#%02x%02x%02x' % (8,31,45))
        fontsize= kwargs.get('fontsize',14)
        figsize = kwargs.get('figsize',(10,4))
        riskLevels = kwargs.get('riskLevels',None)
        xSensor = kwargs.get('xSensor',None)
        offset = kwargs.get('offset',self.info.offset)
        scatterSize = kwargs.get('scatterSize',0.0)
        ax = kwargs.get('ax',None)
        # main plot
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        ax.plot(df['x'].values,df['y'].values,color='k',lw=0.5)
        ax.fill_between(np.array(df['x'].values,float),np.array(df['y'].values,float),float(df['y'].min()),color=groundColor,alpha=0.2)
        # waterLevel
        sections = []
        if level is not None:
            for data in self.get_sections(df,level):
                #ax.hlines(level,data['x'][0],data['x'][-1],color='k',linewidth=0.5)
                ax.fill_between(data['x'],level,data['y'],color=waterColor,alpha=0.9)
                sections.append(data)
        # Sensor
        if (offset is not None) and (xSensor is not None):
            ax.scatter(xSensor,level,marker='v',color='k',s=30+scatterSize,zorder=22)
            ax.scatter(xSensor,level,color='white',s=120+scatterSize+10,edgecolors='k')
            #ax.annotate('nivel actual',xy=(label,level*1.2),fontsize=8)
            #ax.vlines(xSensor, level,offset,linestyles='--',alpha=0.5,color=self.colores_siata[-1])
        #labels
        ax.set_xlabel(xLabel)
        ax.set_facecolor('white')
        #risks
        xlim_max = df['x'].max()
        if riskLevels is not None:
            x = df['x'].max() -df['x'].min()
            y = df['y'].max() -df['y'].min()
            factorx = 0.05
            ancho = x*factorx
            locx = df['x'].max()+ancho/2.0
            miny = df['y'].min()
            locx = 1.03*locx
            risks = np.diff(np.array(list(riskLevels)+[offset]))
            ax.bar(locx,[riskLevels[0]+abs(miny)],width=ancho,bottom=0,color='green')
            colors = ['yellow','orange','red','red']
            for i,risk in enumerate(risks):
                ax.bar(locx,[risk],width=ancho,bottom=riskLevels[i],color=colors[i],zorder=19)

            if level is not None:
                ax.hlines(data['y'].max(),data['x'].max(),locx,lw=1,linestyles='--')
                ax.scatter([locx],[data['y'].max()],s=30,color='k',zorder=20)
            xlim_max=locx+ancho
#        ax.hlines(data['y'].max(),df['x'].min(),sections[0].min(),lw=1,linestyles='--')
        ax.set_xlim(df['x'].min(),xlim_max)

    def in_risk(self,start,end):
        risk = self.risk_level_df(start,end)
        return risk.sum()[risk.sum()<>0.0].index

    @property
    def id_df(self):
        return self.read_sql("select fecha,id from id_hydro where codigo = '%s'"%self.codigo).set_index('fecha')['id']

    def gif_level(self,start,end,delay = 30,loop=0,path = "/media/nicolas/maso/Mario/gifs"):
        level = self.level_local(start,end)
        os.system('rm -r %s/*.png'%path)
        for count in range(level.index.size):
            try:
                nivel = level.copy()
                nivel[count:] = np.NaN
                self.plot_level(nivel,
                                nivel.dropna()[-1]/100.0,
                                figsize=(12,3))
                plt.savefig('%s/%.3d.png'%(path,count),
                            bbox_inches='tight')
                plt.close()
            except:
                pass
        file_name = self.file_format(start,end)+'-gif'
        query = "convert -delay %s -loop %s %s/*.png %s/%s.gif"%(delay,loop,path,path,file_name)
        r = os.system(query)
        if r ==0:
            print('gif saved in path: %s/%s'%(path,file_name))
        else:
            print 'didnt work'

    def plot_level(self,series,lamina='current',resolution='m',legend=True,ax=None,scatter=True,**kwargs):
        '''
        Parameters
        ----------
        series      : level time series pd.Series
        resolution  : list,tuple. Niveles de riesgo
        legend      : risk level legend
        ax          : axis
        scatter     : show level icon, bool
        kwargs : figsize,scatter_size,scatter_color,risk_levels,ymax
        '''
        figsize       = kwargs.get('figsize',(12,4))
        scatter_size  = kwargs.get('scatter_size',0)
        scatter_color = kwargs.get('scatter_color','k')
        risk_levels   = kwargs.get('risk_levels',np.array(self.risk_levels,float)/100.0)
        ymax          = kwargs.get('ymax',max(risk_levels)*1.05)
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        # plot
        ax.fill_between(series.index,series.values,color=self.colores_siata[1],label='Open values')
        # water level
        if series.dropna().index.size == 0:
            x,lamina = (series.index[-1],0.0)
        else:
            if type(lamina) == str:
                if lamina == 'current':
                    x,lamina = (series.dropna().index[-1],series.dropna().iloc[-1])
                else:
                    x,lamina = (series.argmax(),series.max())
        # scatter
        if series.dropna().index.size == 0:
            pass
        else:
            if scatter:
                ax.scatter(x,lamina,marker='v',color=scatter_color,s=30+scatter_size,zorder=22)
                ax.scatter(x,lamina,color='white',s=120+scatter_size+10,edgecolors=scatter_color)
            else:
                pass
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        ax.set_ylabel('Profundidad [%s]'%resolution)
        # legend
        if legend:
            bounds = [(self.loc_time_series(series,0.03)),(self.loc_time_series(series,0.08))]
            ax.axvspan(bounds[0],
                       bounds[1],
                       ymin=0.0,
                       ymax=risk_levels[0]/ymax,
                       color='green')

            ax.axvspan(bounds[0],
                       bounds[1],
                       ymin=risk_levels[0]/ymax,
                       ymax=risk_levels[1]/ymax,
                       color='yellow')

            ax.axvspan(bounds[0],
                       bounds[1],
                       ymin=risk_levels[1]/ymax,
                       ymax=risk_levels[2]/ymax,
                       color='orange')

            ax.axvspan(bounds[0],
                       bounds[1],
                       ymin=risk_levels[2]/ymax,
                       ymax=ymax,
                       color='red')
            ax.set_xlim(series.index[0],bounds[1])
        ax.set_ylim(0,ymax)
        return ax

    def loc_time_series(self,series,percent):
        return series.index[-1]+(series.index[-1]-series.index[0])*percent

    def plot_operacional(self,series,bat,window,filepath):
        '''
        Parameters
        ----------
        series      : level time series pd.Series
        window      : time window, choises are 3h,24h,72h or 30d
        filepath    : path to save file
        '''
        font = {'size'   :25}
        plt.rc('font', **font)
        # figure
        fig  = plt.figure(figsize=(16,24))
        fig.subplots_adjust(hspace=0.8)
        ax = fig.add_subplot(311)
        ax.set_title('Serie de tiempo')
        max_text=True
        if window == '3h':
            lamina = 'current'
        else:
            lamina = 'max'
        try:
            if len(series.dropna()>=1):
                self.plot_level(series,
                                lamina=lamina,
                                risk_levels=np.array(self.risk_levels)/100.0,
                                resolution='m',
                                ax=ax,
                                scatter_size=40)
            else:
                now=pd.datetime.now()
                new_time=pd.date_range(now-datetime.timedelta(hours=int(window.split('h')[0])),now,freq='10T')
                series_aux=pd.Series(np.zeros(len(new_time)),index=new_time)

                self.plot_level(series_aux,
                                lamina=lamina,
                                risk_levels=np.array(self.risk_levels)/100.0,
                                resolution='m',
                                ax=ax,
                                scatter_size=40)
                max_text=False
                ax.set_title('Datos no disponibles para la ventana de tiempo')


            for tick in ax.xaxis.get_major_ticks():
                tick.set_pad( 5.5 * tick.get_pad() )
            # subaxis for xticks in siata format
            box = ax.get_position() #mirror
            subax = fig.add_axes([box.min[0],box.min[1]*0.95, box.width,box.height])
            subax.patch.set_alpha(0.0)
            subs = pd.Series(index=series.index)
            subax.plot(series.index,pd.Series(index=series.index).values,color='w')
            for loc in ['top','right','left']:
                subax.spines[loc].set_visible(False)
            subax.set_xlim(ax.get_xlim())
            # date locator
            major_locator_format = '%d %b %y'
            if window == '3h':
                minor_locator        = mdates.HourLocator(interval=1)
                major_locator        = mdates.DayLocator(interval=1)
            elif window == '24h':
                minor_locator        = mdates.HourLocator(interval=6)
                major_locator        = mdates.DayLocator(interval=1)
            elif window == '72h':
                minor_locator        = mdates.DayLocator(interval=1)
                major_locator        = mdates.DayLocator(interval=1)
            else:
                minor_locator        = mdates.DayLocator(interval=7)
                major_locator        = mdates.DayLocator(interval=7)
            if window !='3h':
                if max_text:
                    ax.annotate(u'máximo', (mdates.date2num(series.argmax()), series.max()), xytext=(10, 10),textcoords='offset points',fontsize=14)
            ax.xaxis.set_major_locator(minor_locator) # notice minor_locator is the max_locator in ax
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            subax.xaxis.set_major_locator(major_locator)
            subax.xaxis.set_major_formatter(mdates.DateFormatter(major_locator_format))
            subax.yaxis.set_major_locator(plt.NullLocator())
        except:
            pass

        # section
        ax2 = fig.add_subplot(312)
        alpha=0.2
        ymax = max([bat['y'].max(),(self.risk_levels[-1])/100.0])

            # plot section
        if series.dropna().index.size == 0:
            lamina = 0.0
        else:
            if lamina == 'current':
                x,lamina = (series.dropna().index[-1],series.dropna().iloc[-1])
            else:
                x,lamina = (series.argmax(),series.max())
        try:
            sections =self.plot_section(bat,
                                    ax = ax2,
                                    level=lamina,
                                    riskLevels=np.array(self.risk_levels)/100.0,
                                    xSensor=self.info.x_sensor,
                                    scatterSize=50)
        except:
            pass
        ax2.spines['top'].set_color('w')
        ax2.spines['right'].set_color('w')
        ax2.spines['right'].set_color('w')
        ax2.set_ylim(bat['y'].min(),ymax)
        ax2.set_title('Profundidad en el canal')
        ax2.set_ylabel('Profundidad [m]')
        # ax3 RESUMEN
        try:
            format =   (['ultrasonido','radar'][self.info.tipo_sensor],
                        series.dropna().index.size*100.0/series.index.size,
                        series.max(),
                        ['verde','amarillo','naranja','rojo','rojo'][self.convert_level_to_risk(series.max(),np.array(self.risk_levels)/100.0)],
                        series.mean())
            text= u'-Estación de Nivel tipo %s\n-Resolución temporal: 1 minutos\n-%% de datos transmitidos: %.2f\n-Profundidad máxima: %.2f [m]\n-Nivel de riesgo máximo: %s\n-Profundidad promedio: %.2f [m]\n*Calidad de datos a\xfan sin \n verificar exhaustivamente'%(format)
        except:
            text = u'ESTACIÓN SIN DATOS TEMPORALMENTE'

        ax4=fig.add_subplot(413)
        img=plt.imread('/media/nicolas/Home/Jupyter/Sebastian/git/CPR/cprv1/leyenda.png')
        im=ax4.imshow(img)
        pos=im.axes.get_position()
        im.axes.set_position((pos.x0-.026,pos.y0-.17,pos.x1-.026,pos.y1-.17))
        ax4.axis('off')

        ax3 = fig.add_subplot(414)
        ax3.text(0.0,1.1,'RESUMEN',color = self.colores_siata[-1])
        ax3.text(0.0, 0.0,text,linespacing=2.1,fontsize=15)
        plt.axis('off')
        plt.suptitle('%s | %s'%(self.codigo,self.info.nombre),y=0.93)
        plt.savefig(filepath,bbox_inches='tight')
        return ax,ax2,ax3

    def update_level_local(self,start,end):
        self.table = 'hydro'
        try:
            s = self.sensor(start,end).resample('5min').mean()
            self.update_series(s,'nivel')
        except:
            print 'WARNING: No data for %s'%self.codigo

    def update_level_local_all(self,start,end):
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        timer = datetime.datetime.now()
        size = self.infost.index.size
        for count,codigo in enumerate(self.infost.index):
            obj = Nivel(codigo = codigo,SimuBasin=False,**info.LOCAL)
            obj.table = 'hydro'
            print "%s out of %s | %s"%(count+1,size,obj.info.nombre)
            obj.update_level_local(start,end)
        seconds = (datetime.datetime.now()-timer).seconds
        print 'Full updating took %s minutes'%(seconds/60.0)

    def calidad(self):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=7)
        df = self.read_sql("select fecha,nivel,codigo from hydro where fecha between '%s' and '%s'"%(start.strftime('%Y-%m-%d %H:%M'),end.strftime('%Y-%m-%d %H:%M')))
        now = datetime.datetime.now()
        s = pd.DataFrame(df.loc[df.nivel.notnull()].groupby('codigo')['fecha'].max().sort_values())
        s['nombre'] = self.infost.loc[s.index,'nombre']
        s['delta'] = now-s['fecha']
        for horas,valor in zip([1,3,24,72],['green','yellow','orange','red']):
            r = s['fecha']<(now-datetime.timedelta(hours=horas))
            s.loc[s[r].index,'rango']=valor
        return s.dropna()

    def reporte_calidad(self,path):
        df = self.calidad()
        df = df[['nombre','fecha','delta','rango']]
        df = df.reset_index()
        s = df['rango']
        df = df.drop('rango',axis=1)
        lista = []
        for id in df.index:
            lista.append(list(df.loc[id]))
        doc = SimpleDocTemplate(path, pagesize=(520,600))
        # container for the 'Flowable' objects
        elements = []
        style = ParagraphStyle(
            name='Normal',
            aligment = TA_LEFT)

        elements.append(Paragraph("Reporte de estaciones caidas", style))
        lista.insert(0,['Código','Nombre','Último dato','Delta'])
        data = lista
        #data = df.applyvalues
        t=Table(data)
        for pos,color in zip(s.index,s.values):
            t.setStyle(TableStyle([('BACKGROUND',(0,pos+1),(0,pos+1),color)]))
        elements.append(t)

        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),self.colores_siata[-1]),
                               ('TEXTCOLOR',(0,0),(-1,0),'white'),
                               ('ALIGN',(0,0),(-1,0),'CENTER')]))
        # write the document to disk
        doc.build(elements)
        #doc.drawString(200,5000,u'Nivel sin riesgo')

    def add_area_metropol(self,m):
        m.readshapefile('/media/nicolas/maso/Mario/shapes/AreaMetropolitana','area',linewidth=0.5,color='w')
        x,y = m(self.info.longitud,self.info.latitud)
        #m.scatter(x,y,s=100,zorder=10)
        scatterSize=100
        m.scatter(x,y,color='grey',s=120+scatterSize+60,edgecolors='grey',zorder=39)
        m.scatter(x,y,color='w',s=120+scatterSize+60,edgecolors='k',zorder=40)
        m.scatter(x,y,marker='v',color='k',s=20+scatterSize,zorder=41)
        municipios = [m.area_info[i]['Name'] for i in range(10)]
        patches=[]
        for info,shape in zip(m.area_info,m.area):
            patches.append(Polygon(np.array(shape),True),)
        plt.gca().add_collection(PatchCollection(patches,color='grey',edgecolor='w',zorder=1,alpha=0.3,label='asdf'))
        for frame in ['top','bottom','right','left']:
            plt.gca().spines[frame].set_color('w')

    def plot_rain_future(self,current_vect,future_vect,filepath=None):
        plt.close('all')
        fig = plt.figure(figsize=(16,8))
        fig.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=0.1,
                            hspace=None)
        ax1 = fig.add_subplot(121)
        m = self.plot_basin_rain(current_vect,ax=ax1)
        self.add_area_metropol(m)
        ax2 = fig.add_subplot(122)
        m = self.plot_basin_rain(future_vect,ax=ax2,cbar=True)
        self.add_area_metropol(m)
        if filepath:
            plt.savefig(filepath,bbox_inches='tight')

    def plot_level_report(self,level,rain,riesgos,fontsize=14,ncol=4,ax=None,bbox_to_anchor=(1.0,1.2),**kwargs):
        if ax is None:
            fig = plt.figure(figsize=(13.,4))
            ax = fig.add_subplot(111)
        #nivel = level.resample('H',how='mean')
        #nivel.plot(ax=ax,label='',color='k')
        mean_rain = rain
        ax.set_xlim(level.index[0],mean_rain.index[-1])
        level.plot(label='',color='k',fontsize=fontsize,lw=2,zorder=20,**kwargs)
        axu= ax.twinx()
        axu.set_ylabel('Intensidad promedio [mm/h]',fontsize=fontsize)
        mean_rain.plot(ax=axu,alpha=0.5,fontsize=fontsize,**kwargs)
        axu.fill_between(mean_rain.index,0,mean_rain.values,alpha=0.2)
        ylim = axu.get_ylim()[::-1]
        ylim = (ylim[0],0.0)
        axu.set_ylim(ylim)
        ax.set_ylabel('Profundidad (cm)',fontsize=fontsize)
        ax.set_ylim(0,riesgos[-1]*1.2)
        alpha=0.9
        ax.fill_between(level.index[:3],ax.get_ylim()[0],riesgos[0],alpha=alpha,color='g')
        ax.fill_between(level.index[:3],riesgos[0],riesgos[1],alpha=alpha,color='yellow')
        ax.fill_between(level.index[:3],riesgos[1],riesgos[2],alpha=alpha,color='orange')
        #ax.fill_between(nivel.index,riesgos[2],riesgos[3],alpha=alpha,color='red')
        ax.fill_between(level.index[:3],riesgos[2],ax.get_ylim()[1],alpha=alpha,color='red')
        scatterSize=100
        ax.hlines(level.loc[level.index[-1]],level.index[0],level.index[-1],linewidth=1.0,linestyles='dashed')
        ax.scatter(level.index[-1],level.loc[level.index[-1]],color='grey',s=120+scatterSize+60,edgecolors='grey',zorder=39)
        ax.scatter(level.index[-1],level.loc[level.index[-1]],color='w',s=120+scatterSize+60,edgecolors='k',zorder=40)
        ax.scatter(level.index[-1],level.loc[level.index[-1]],marker='v',color='k',s=20+scatterSize,zorder=41)

    def rain_report(self,date):
        date = pd.to_datetime(date)
        start = date-datetime.timedelta(minutes=150)# 3 horas atras
        end = date+datetime.timedelta(minutes=30)
        #filepaths
        local_path = '/media/nicolas/Home/Jupyter/MarioLoco/reportes_lluvia/'
        remote_path = 'mcano@siata.gov.co:/var/www/mario/reportes_lluvia/'
        day_path = local_path + date.strftime('%Y%m%d')+'/'
        station_path = day_path+str(self.codigo)+'/'
        filepath = station_path+self.file_format(start,end)
        #make directories
        os.system('mkdir %s'%day_path)
        os.system('mkdir %s'%station_path)
        #rain
        vec = self.radar_rain(start,end,ext='.bin')
        current_vect = vec.drop(vec.loc[date:].index).sum().values/1000
        future_vect = vec.drop(vec.loc[:date].index).sum().values/1000
        # level
        mean_rain = self.radar_rain(start,end)*12.0
        series = self.level(start,date).resample('5min').mean()
        series[series>self.info.offset] = np.NaN
        series.index.name = ''
        level_cond = (series.dropna().size/series.size) < 0.05 # condición de nivel para graficar
        rain_cond = len(current_vect)==0.0
        if level_cond:
            print 'Not enough level data'
        if rain_cond:
            print 'Not rain in basin'
        if level_cond or rain_cond:
            pass
        else:
            #plots
            self.plot_rain_future(current_vect,future_vect,filepath = filepath+'_rain')
            self.plot_level_report(series,mean_rain,self.risk_levels)
            plt.gca().set_xlim(start,end)
            plt.savefig(filepath+'_level.png',bbox_inches='tight')
            self.rain_report_reportlab(filepath,date)
            os.system('ssh mcano@siata.gov.co "mkdir /var/www/mario/reportes_lluvia/%s"'%(date.strftime('%Y%m%d')))
            query = "rsync -r %s %s/%s/"%(filepath+'_report.pdf',remote_path+date.strftime('%Y%m%d'),self.codigo)
            os.system(query)

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
        pdf.showPage()
        pdf.save()

    def level_local_all(self,start,end):
        start,end = (start.strftime('%Y-%m-%d %H:%M'),end.strftime('%Y-%m-%d %H:%M'))
        query = "select codigo,fecha,nivel from hydro where fecha between '%s' and '%s'"%(start,end)
        df = self.read_sql(query).set_index('codigo').loc[self.infost.index].set_index('fecha',append=True)
        codigos = df.index.levels[0]
        nivel = df.reset_index('fecha').loc[codigos,'nivel']
        df = df.reset_index('fecha')
        df['nivel'] = self.infost.loc[df.index,'offset']-df['nivel']
        df = df.set_index('fecha',append=True)
        df[df<0.0] = np.NaN
        return df.unstack(0)['nivel']

    def make_rain_report_current(self,codigos):
        for codigo in codigos:
            nivel = cpr.Nivel(codigo = codigo,SimuBasin=True,**info.LOCAL)
            nivel.rain_report(datetime.datetime.now())

    def risk_df(self,df):
        for codigo in df.columns:
            risk_levels = np.array(self.infost.loc[codigo,['n1','n2','n3','n4']])
            try:
                df[codigo] = df[codigo].apply(lambda x:self.convert_level_to_risk(x,risk_levels))
            except:
                df[codigo] = np.NaN
        df = df[df.sum().sort_values(ascending=False).index].T
        return df

    def make_risk_report(self,df,figsize=(6,14),bbox_to_anchor = (-0.15, 1.09),ruteSave = None,legend=True):
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle
        def make_colormap(seq):
            seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
            cdict = {'red': [], 'green': [], 'blue': []}
            for i, item in enumerate(seq):
                if isinstance(item, float):
                    r1, g1, b1 = seq[i - 1]
                    r2, g2, b2 = seq[i + 1]
                    cdict['red'].append([item, r1, r2])
                    cdict['green'].append([item, g1, g2])
                    cdict['blue'].append([item, b1, b2])
            return mcolors.LinearSegmentedColormap('CustomMap', cdict)
        df = df.loc[df.index[::-1]]
        c = mcolors.ColorConverter().to_rgb
        cm = make_colormap([c('green'),0.20,c('#f2e71d'),0.4,c('orange'),0.60,c('red'),0.80,c('red')])
        fig = plt.figure(figsize=figsize)
        im = plt.imshow(df.values, interpolation='nearest', vmin=0, vmax=4, aspect='equal',cmap=cm);
        #cbar = fig.colorbar(im)
        ax = plt.gca();
        ax.set_xticks(np.arange(0,df.columns.size, 1));
        ax.set_yticks(np.arange(0, df.index.size, 1));
        ax.set_xticklabels(df.columns,fontsize=14);
        ax.set_yticklabels(df.index,fontsize=14,ha = 'left');
        ax.set_xticks(np.arange(-.5, df.columns.size, 1), minor=True,);
        ax.set_yticks(np.arange(-.5, df.index.size, 1), minor=True);
        plt.draw()
        yax = ax.get_yaxis()
        pad = max(T.label.get_window_extent().width*1.05 for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        #ax.Axes.tick_params(axis='x', rotation=45)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        plt.xticks(rotation=90)
        alpha=1
        height = 8

    def level_all(self,start=None,end = None,hours=3,**kwargs):
        if start:
            start = self.round_time(pd.to_datetime(start))
            end   = self.round_time(pd.to_datetime(end))
        else:
            end = pd.to_datetime(self.round_time())
            start = end - datetime.timedelta(hours = hours)
        codigos = kwargs.get('codigos',self.infost.index)
        df = pd.DataFrame(index = pd.date_range(start,end,freq='5min'),columns = codigos)
        for codigo in codigos:
            try:
                level = Nivel(codigo=codigo,** info.LOCAL).level(start,end,**kwargs).resample('5min').mean()
                df[codigo] = level
            except:
                pass
        return df

    def make_risk_report_current(self,df):
        # estaciones en riesgo
        df = df.copy()
        in_risk = df.T
        in_risk = in_risk.sum()[in_risk.sum()!=0.0].index.values
        df.columns = map(lambda x:x.strftime('%H:%M'),df.columns)
        df.index = np.array(df.index.values,str)+(np.array([' | ']*df.index.size)+self.infost.loc[df.index,'nombre'].values)
        self.make_risk_report(df,figsize=(15,25))
        filepath = '/media/nicolas/Home/Jupyter/MarioLoco/reportes/reporte_niveles_riesgo_actuales.png'
        plt.savefig(filepath,bbox_inches='tight')
        os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/'%filepath)
        return in_risk

    def reporte_nivel(self):
        def convert_to_risk(df):
            df = self.risk_df(df)
            return df[df.columns.dropna()]
        self.make_risk_report_current(convert_to_risk(self.level_all()))

    def rain_area_metropol(self,vec,ax,f=1):
        cmap_radar,levels,norm = self.radar_cmap()
        extra_lat,extra_long = self.adjust_basin(fac=0.02)
        extra_long=0
        extra_lat=0
        kwargs = {}
        contour_keys={'cmap'  :cmap_radar,
                    'levels':levels,
                    'norm'  :norm}
        perimeter_keys={'color':'k','linewidth':1}
        longs,lats=self.longitude_latitude_basin()
        x,y=np.meshgrid(longs,lats)
        y=y[::-1]
        # map settings
        m = Basemap(projection='merc',llcrnrlat=lats.min()-0.05*f, urcrnrlat=lats.max()+0.05*f,
        llcrnrlon=longs.min()-0.05*f, urcrnrlon=longs.max()+0.1*f, resolution='c',ax=ax,**kwargs)
        # perimeter plot
        xp,yp = m(self.Polygon[0], self.Polygon[1])
        m.plot(xp, yp,**perimeter_keys)
        # vector plot
        if vec is not None:
            map_vec,mxll,myll=wmf.cu.basin_2map(self.structure,vec,len(longs),len(lats),self.ncells)
            map_vec[map_vec==wmf.cu.nodata]=np.nan
            xm,ym=m(x,y)
            contour = m.contourf(xm, ym, map_vec.T, 25,**contour_keys)
        else:
            contour = None
        m.readshapefile('/media/nicolas/maso/Mario/shapes/AreaMetropolitana','area',linewidth=0.5,color='w')
        m.readshapefile('/media/nicolas/maso/Mario/shapes/net/%s/%s'%(self.codigo,self.codigo),str(self.codigo))
        m.readshapefile('/media/nicolas/maso/Mario/shapes/streams/%s/%s'%(self.codigo,self.codigo),str(self.codigo))
        x,y = m(self.info.longitud,self.info.latitud)
        #m.scatter(x,y,s=100,zorder=10)
        scatterSize=100
        m.scatter(x,y,color='grey',s=120+scatterSize+60,edgecolors='grey',zorder=39)
        m.scatter(x,y,color='w',s=120+scatterSize+60,edgecolors='k',zorder=40)
        m.scatter(x,y,marker='v',color='k',s=20+scatterSize,zorder=41)
        municipios = [m.area_info[i]['Name'] for i in range(10)]
        patches=[]
        for info,shape in zip(m.area_info,m.area):
            patches.append(Polygon(np.array(shape),True),)
        ax.add_collection(PatchCollection(patches,color='grey',edgecolor='w',zorder=1,alpha=0.3,label='asdf'))
        #m.readshapefile('/media/nicolas/maso/Mario/shapes/polygon/145/145','sabanetica',zorder=100)
        for frame in ['top','bottom','right','left']:
            ax.spines[frame].set_color('w')
        cbar = m.colorbar(contour,location='right',pad="5%")

    def convert_series_to_risk(self,level):
        '''level: pandas Series, index = codigos de estaciones'''
        risk = level.copy()
        colors = ['green','gold','orange','red','red','black']
        for codigo in level.index:
            try:
                risks = cpr.Nivel(codigo = codigo,**info.LOCAL).risk_levels
                risk[codigo] = colors[int(self.convert_level_to_risk(level[codigo],risks))]
            except:
                risk[codigo] = 'black'
        return risk

    def reporte_lluvia(self,end,filepath=None):
            self = Nivel(codigo=260,SimuBasin=True,**info.LOCAL)
            #end = datetime.datetime.now()
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
            risk = self.convert_series_to_risk(self.level_all(hours=1).iloc[-3:].max())
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
            if filepath:
                plt.savefig(filepath,bbox_inches='tight')
            #os.system('scp %s mcano@siata.gov.co:/var/www/mario/realTime/reporte_lluvia_cuenca.png'%filepath)

    def plot_risk_daily(self,df,bbox_to_anchor = (-0.15, 1.09),figsize=(6,14),ruteSave = None,legend=True,fontsize=20):
        import matplotlib.colors as mcolors
        def make_colormap(seq):
            seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
            cdict = {'red': [], 'green': [], 'blue': []}
            for i, item in enumerate(seq):
                if isinstance(item, float):
                    r1, g1, b1 = seq[i - 1]
                    r2, g2, b2 = seq[i + 1]
                    cdict['red'].append([item, r1, r2])
                    cdict['green'].append([item, g1, g2])
                    cdict['blue'].append([item, b1, b2])
            return mcolors.LinearSegmentedColormap('CustomMap', cdict)
        df = df.loc[df.index[::-1]]
        c = mcolors.ColorConverter().to_rgb
        cm = make_colormap([c('green'),0.20,c('#f2e71d'),0.4,c('orange'),0.60,c('red'),0.80,c('red')])
        fig = plt.figure(figsize=figsize)
        im = plt.imshow(df.values, interpolation='nearest', vmin=0, vmax=4, aspect='equal',cmap=cm);
        #cbar = fig.colorbar(im)
        ax = plt.gca();
        ax.set_xticks(np.arange(0,df.columns.size, 1));
        ax.set_yticks(np.arange(0, df.index.size, 1));
        ax.set_xticklabels(df.columns,fontsize=fontsize);
        ax.set_yticklabels(df.index,fontsize=fontsize,ha = 'left');
        ax.set_xticks(np.arange(-.5, df.columns.size, 1), minor=True,);
        ax.set_yticks(np.arange(-.5, df.index.size, 1), minor=True);
        plt.draw()
        yax = ax.get_yaxis()
        pad = max(T.label.get_window_extent().width*1.05 for T in yax.majorTicks)
        yax.set_tick_params(pad=pad)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        #ax.text(-0.4,df.index.size+0.5,'NIVELES DE RIESGO\n %s - %s'%(start,pd.to_datetime(end).strftime('%Y-%m-%d')),fontsize=16)
        alpha=1
        height = 8
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

    def reporte_diario(self,date):
        end = pd.to_datetime(pd.to_datetime(date).strftime('%Y-%m-%d')+' 23:55') - datetime.timedelta(days=1)
        start = (end-datetime.timedelta(days=6)).strftime('%Y-%m-%d 00:00')
        folder_path = '/media/nicolas/Home/Jupyter/MarioLoco/reporte_diario/%s'%end.strftime('%Y%m%d')
        os.system('mkdir %s'%folder_path)
        df = self.level_all(start,end)
        from matplotlib.patches import Rectangle
        try:
            df = df.T.drop([1013,1014,195,196]).T
        except:
            pass
        daily = df.resample('D').max()
        rdf = self.risk_df(daily)
        # niveles de riesgo en el último día
        last_day_risk = rdf[rdf.columns[-1]].copy()
        last_day_risk = last_day_risk[last_day_risk>0.0].sort_values(ascending=False).index
        rdf = rdf.loc[rdf.max(axis=1).sort_values(ascending=False).index]
        rdf = rdf[rdf.max(axis=1)>0.0]
        rdf = rdf.fillna(0)
        labels = []
        for codigo,nombre in zip(self.infost.loc[rdf.index].index,self.infost.loc[rdf.index,'nombre'].values):
            labels.append('%s | %s'%(codigo,nombre))
        rdf.index = labels
        def to_col_format(date):
            return (['L','M','MI','J','V','S','D'][int(date.strftime('%u'))-1]+date.strftime('%d'))
        rdf.columns = map(lambda x:to_col_format(x),rdf.columns)
        import sys
        # sys.setdefaultencoding() does not exist, here!
        reload(sys)  # Reload does the trick!
        sys.setdefaultencoding('UTF8')
        self.plot_risk_daily(rdf,figsize=(14,20))
        plt.savefig(folder_path+'/reporte_nivel.png',bbox_inches='tight')
        remote_path = 'mcano@siata.gov.co:/var/www/mario/reporte_diario/'
        query = "rsync -r %s %s/"%(folder_path+'/reporte_nivel.png',remote_path+end.strftime('%Y%m%d'))
        os.system(query)

        #Graficas
        fontsize = 25
        font = {'size'   :fontsize}
        plt.rc('font', **font)
        filepath = None

        for num,codigo in enumerate(np.array(last_day_risk,int)):
            obj = Nivel(codigo=codigo,SimuBasin=False,**info.LOCAL)
            series = df.loc[daily.index[-1].strftime('%Y-%m-%d'),codigo]
            series = obj.level(series.index[0],series.index[-1])
            if series.dropna().index.size==0.0:
                pass
            else:
                plt.figure()
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(25,4))
                obj.plot_level(series/100.0,
                                lamina='max',
                                risk_levels=np.array(obj.risk_levels)/100.0,
                                legend=False,
                                resolution='m',
                                ax=ax1,
                                scatter_size=40)

                alpha=0.2
                bat = obj.last_bat(obj.info.x_sensor)
                ymax = max([bat['y'].max(),(obj.risk_levels[-1])/100.0])
                lamina = 'max'
                    # plot section
                if series.dropna().index.size == 0:
                    lamina = 0.0
                else:
                    if lamina == 'current':
                        x,lamina = (series.dropna().index[-1],series.dropna().iloc[-1])
                    else:
                        x,lamina = (series.argmax(),series.max())

                sections =obj.plot_section(bat,
                                        ax = ax2,
                                        level=lamina/100.0,
                                        riskLevels=np.array(obj.risk_levels)/100.0,
                                        xSensor=obj.info.x_sensor,
                                        scatterSize=50)
                major_locator        = mdates.DayLocator(interval=5)
                formater = '%H:%M'
                ax1.xaxis.set_major_formatter(mdates.DateFormatter(formater))
                ax1.set_xlabel(u'Fecha')
                ax2.spines['top'].set_color('w')
                ax2.spines['right'].set_color('w')
                ax2.spines['right'].set_color('w')
                ax2.set_ylim(bat['y'].min(),ymax)
                ax1.set_ylim(bat['y'].min(),ymax)
                ax1.set_title(u'código: %s'%codigo)
                ax2.set_title('Profundidad en el canal')
                ax2.set_ylabel('Profundidad [m]')
                #ax1.set_xlabel('03 Mayo - 04 Mayo')
                ax1.annotate(u'máximo', (mdates.date2num(series.argmax()), series.max()/100.0), xytext=(10, 10),textcoords='offset points',fontsize=fontsize)
                #file = 'section_%s.png'%(num+1)
                ax2.set_title(obj.info.nombre)
                #filepath = 'reportes_amva/%s.png'%codigo
                for tick in ax1.get_xticklabels():
                    tick.set_rotation(45)
                filepath = folder_path+'/'+obj.info.slug+'.png'
                plt.savefig(filepath,bbox_inches='tight')
                os.system('rsync %s %s'%(filepath,remote_path+end.strftime('%Y%m%d')+'/'))

        obj = Nivel(codigo=260,SimuBasin=True,**info.LOCAL)
        radar_rain = obj.radar_rain_vect(start,end)
        diario = radar_rain.resample('D').sum()
        rain = obj.radar_rain(start,end)
        fig = plt.figure(figsize=(20,20))
        for pos,dia in enumerate(diario.index):
            ax = fig.add_subplot(3,3,pos+1)
            obj.rain_area_metropol(diario.loc[dia].values/1000.0,ax)
            ax.set
            plt.gca().set_title(rdf.columns[pos])
        plt.savefig(folder_path+'/lluvia_diaria.png',bbox_inches='tight')
        remote_path = 'mcano@siata.gov.co:/var/www/mario/reporte_diario/'
        query = "rsync -r %s %s/"%(folder_path+'/lluvia_diaria.png',remote_path+end.strftime('%Y%m%d'))
        os.system(query)

    def gif(self,start,end,delay=0,loop=0):
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        rain_vect = self.radar_rain_vect(start,end)
        rain = self.radar_rain(start,end)*12.0
        bat = self.last_bat(self.info.x_sensor)
        nivel = self.level(start,end).resample('5min',how='mean')
        rain_vect = rain_vect.reindex(nivel.index)
        rain = rain.reindex(nivel.index)
        # plot gif function before loop
        filepath = '/media/nicolas/maso/Mario/user_output/gifs/%s/'%self.file_format(start,end)
        os.system('mkdir %s'%filepath)
        def plot_gif(count,fecha,f=1,filepath=filepath,**kwargs):
            fontsize = 18
            font = {'size'   :fontsize}
            plt.rc('font', **font)
            series = nivel.copy()
            series[count:] = np.NaN
            vect = rain_vect.copy()
            vect = vect.drop(vect.index[count:])
            s = rain.copy()
            s[count:] = np.NaN
            level = series.dropna().iloc[-1]/100.0
            vec = vect.sum().values/1000.0
            scatterSize = 70
            figsize=(18,14)
            series = pd.Series.copy(series/100.0)
            risk_levels = np.array(self.risk_levels,float)/100.0
            fig = plt.figure(figsize=figsize)
            fig.subplots_adjust(wspace=0.3)
            #gs = GridSpec(3, 3)
            ax1 = fig.add_subplot(2,2,1)
            # identical to ax1 = plt.subplot(gs.new_subplotspec((0,0), colspan=3))
            ax2 = fig.add_subplot(2,2,2,sharey=ax1)
            ylimit = kwargs.get('ylimit',max(risk_levels)*1.05)
            series.plot(ax=ax1,label='',color='w',linewidth=0.001,fontsize=fontsize,**kwargs)
            #ax1.fill_between(series.index,series.values,color=self.colores_siata[0])
            ax1.fill_between(series.index, series.values,color=self.colores_siata[1],label='Open values')
            #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d %b'))
            alpha=0.2
            ymax = max([bat['y'].max(),risk_levels[-1]])
            ax1.set_xlim(series.index[0],series.index[-1]+datetime.timedelta(minutes=5))
            sections =self.plot_section(bat,
                                   ax = ax2,
                                   level=level,
                                   riskLevels=risk_levels,
                                   xSensor=self.info.x_sensor,
                                   scatterSize=scatterSize)
            ax1.spines['top'].set_color('w')
            ax1.spines['right'].set_color('w')
            ax2.spines['top'].set_color('w')
            ax2.spines['right'].set_color('w')
            ax2.spines['right'].set_color('w')
            ax1.set_ylabel('Profundidad [m]')
            ax1.set_xlim(rain.index[0],rain.index[-1])
            ax1.scatter(series.dropna().index[-1],level,marker='v',color='k',s=30+scatterSize,zorder=22)
            ax1.scatter(series.dropna().index[-1],level,color='white',s=120+scatterSize+10,edgecolors='k')
            ax3 = fig.add_subplot(2,2,3,sharex=ax1)
            s.plot(ax=ax3,color='w',linewidth=0.001,fontsize=fontsize)
            ax3.fill_between(s.index, s.values,color=self.colores_siata[3],label='O')
            ax3.spines['top'].set_color('w')
            ax3.spines['right'].set_color('w')
            ax3.spines['right'].set_color('w')
            ax4 = fig.add_subplot(2,2,4)
            self.rain_area_metropol(vec,ax4,f=f)
            ax2.set_ylim(0,ymax)
            #cb = plt.colorbar(ax, cax = cbaxes)
            #cb.set_level('[mm]')
            ax1.set_title(u'Profundidad de la lámina de agua')
            ax2.set_title(u'Sección transversal del canal')
            ax3.set_title(u'Lluvia promedio en la cuenca')
            ax3.set_ylabel('Intensidad promedio [mm/h]',fontsize=fontsize)
            ax4.set_title(u'Lluvia acumulada [mm]')
            ax3.set_ylim(0,rain.max())
            ax3.set_xlim(rain.index[0],rain.index[-1])
            plt.suptitle(u'%s | %s'%(self.info.nombre,fecha))
            path = '%s%.3d.png'%(filepath,count)
            print path
            plt.savefig(path,bbox_inches='tight')
        # loop
        for count in range(1,rain.index.size+1):
                plot_gif(count,self.info.nombre,f=0.5)

        file_name = self.file_format(start,end)+'-gif'
        query = "convert -delay %s -loop %s %s*.png %s%s.gif"%(delay,loop,filepath,filepath,file_name)
        r = os.system(query)
        r=0
        if r ==0:
            print('gif saved in path: %s%s'%(filepath,file_name))
        else:

            print 'didnt work'
        filepath = filepath+file_name
        remote_path = 'mcano@siata.gov.co:/var/www/mario/gifs/'
        os.system('ssh mcano@siata.gov.co "mkdir /var/www/mario/gifs/%s"'%(end.strftime('%Y%m%d')))
        query = "rsync -r %s.gif %s/%s/"%(filepath,remote_path+end.strftime('%Y%m%d'),self.codigo)
        return os.system(query)

class RedRio(Nivel):
    def __init__(self,**kwargs):
        Nivel.__init__(self,**kwargs)
        self.info = pd.Series(index=Nivel(codigo=99,user='mario',passwd='mc@n0Yw2E').info.copy().index)
        self.info.slug = self.info_redrio.loc[self.codigo,'Nombre'].decode('utf8').encode('ascii', errors='ignore').replace('.','').replace(' ','').replace('(','').replace(')','')
        self.fecha = '2006-06-06 06:06'
        self.workspace = '/media/nicolas/Home/Jupyter/MarioLoco/repositories/CPR/documentacion/redrio/'
        self.seccion = pd.DataFrame(columns = [u'vertical', u'x', u'y', u'v01', u'v02', u'v03', u'v04', u'v05', u'v06', u'v07', u'v08', u'v09', u'vsup'])
        self.parametros = "id_aforo,fecha,ancho_superficial,caudal_medio,velocidad_media,perimetro,area_total,altura_media,radio_hidraulico"
        self.aforo = pd.Series(index = [u'fecha', u'ancho_superficial', u'caudal_medio', u'velocidad_media',u'perimetro', u'area_total',
     u'altura_media', u'radio_hidraulico',u'levantamiento'])
        self.info.nc_path = '/media/nicolas/maso/Mario/basins/%s.nc'%self.codigo
        self.info.nombre = self.info_redrio.loc[self.codigo,'Nombre']
        self.info.longitud = self.info_redrio.loc[self.codigo,'Longitud']
        self.info.latitud = self.info_redrio.loc[self.codigo,'Latitud']
        self.levantamiento = pd.DataFrame(columns = ['vertical','x','y'])
        self.alturas = pd.DataFrame(index=pd.date_range(start = pd.to_datetime('2018').strftime('%Y-%m-%d 06:00'),periods=13,freq='H'),columns = ['profundidad','offset','lamina','caudal'])
        self.alturas.index = map(lambda x:x.strftime('%H:00'),self.alturas.index)

    @property
    def info_redrio(self):
        return pd.read_csv('/media/nicolas/Home/Jupyter/MarioLoco/redrio/info_redrio.csv',index_col=0)

    @property
    def caudales(self):
        return self.aforos().set_index('fecha')['caudal']

    @property
    def folder_path(self):
        return self.workspace+pd.to_datetime(self.fecha).strftime('%Y%m%d')+'/'+self.info.slug+'/'

    @property
    def aforo_nueva(self):
        pass

    @property
    def seccion_aforo_nueva(self):
        pass

    @property
    def levantamiento_aforo_nueva(self):
        pass

    def get_levantamiento(self,id_aforo):
        seccion = self.read_sql("SELECT * FROM levantamiento_aforo_nueva WHERE id_aforo = '%s'"%(id_aforo)).set_index('vertical')
        return seccion[['x','y']].sort_index()

    def aforos(self,filter=True):
        aforos = self.read_sql("SELECT %s from aforo_nueva where id_estacion_asociada = '%s'"%(self.parametros,self.codigo))
        aforos = aforos.set_index('id_aforo')
        if filter:
            aforos[aforos==-999]=np.NaN
        aforos = aforos.dropna()
        aforos = aforos.sort_values('fecha')
        aforos['levantamiento']=False
        for id_aforo in aforos.index:
            if self.get_levantamiento(id_aforo).index.size:
                aforos.loc[id_aforo,'levantamiento'] = True
        return aforos
    @property
    def levantamientos(self):
        return self.aforos(filter=False)[self.aforos(filter=False)['levantamiento']].index

    def insert_vel(self,vertical,v02,v04,v08):
        self.seccion.loc[vertical,'v02'] = v02
        self.seccion.loc[vertical,'v04'] = v04
        self.seccion.loc[vertical,'v08'] = v08

    def velocidad_media_dovela(self):
        columns = [u'vertical', u'x', u'y', u'v01', u'v02', u'v03', u'v04', u'v05', u'v06', u'v07', u'v08', u'v09', u'vsup']
        dfs = self.seccion[columns].copy()
        self.seccion['vm'] = np.NaN
        vm = []
        for index in dfs.index:
            vm.append(round(self.estima_velocidad_media_vertical(dfs.loc[index].dropna()),3))
        self.seccion['vm'] = vm
    def area_dovela(self):
        self.seccion['area'] = self.get_area(self.seccion['x'].abs().values,self.seccion['y'].abs().values)

    def estima_velocidad_media_vertical(self,vertical,factor=0.0,v_index=0.8):
        vertical = vertical[vertical.index!='vm']
        index = list(vertical.index)
        if index == ['vertical','x','y']:
            if vertical['x'] == 0.0:
                vm = factor * self.seccion.loc[vertical.name+1,'vm']
            else:
                vm = factor * self.seccion.loc[vertical.name-1,'vm']
        elif (index == ['vertical','x','y','vsup']) or (index == ['vertical','x','y','v08']):
            try:
                vm = v_index*vertical['vsup']
            except:
                vm = v_index*vertical['v08']
        elif (index == ['vertical','x','y','v04']) or (index == ['vertical','x','y','v04','vsup']):
            vm = vertical['v04']
        elif index == (['vertical','x','y','v04','v08']) or index == (['vertical','x','y','v04','v08','vsup'])  :
            vm = (2*vertical['v04']+vertical['v08'])/3.0
        elif index == ['vertical','x','y','v08','vsup']:
            vm = v_index*vertical['vsup']
        elif (index == ['vertical','x','y','v02','v04','v08']) or (index == ['vertical','x','y','v02','v04','v08','vsup']):
            vm = (2*vertical['v04']+vertical['v08']+vertical['v02'])/4.0
        return vm

    def perimetro(self):
        x,y = (self.seccion['x'].values,self.seccion['y'].values)
        def perimetro(x,y):
            p = []
            for i in range(len(x)-1):
                p.append(float(np.sqrt(abs(x[i]-x[i+1])**2.0+abs(y[i]-y[i+1])**2.0)))
            return [0]+p
        self.seccion['perimetro'] = perimetro(self.seccion['x'].values,self.seccion['y'].values)

    def get_area(self,x,y):
        '''Calcula las áreas y los caudales de cada
        una de las verticales, con el método de mid-section
        Input:
        x = Distancia desde la banca izquierda, type = numpy array
        y = Produndidad
        v = Velocidad en la vertical
        Output:
        area = Área de la subsección
        Q = Caudal de la subsección
        '''
        # cálculo de áreas
        d = np.absolute(np.diff(x))/2.
        b = x[:-1]+d
        area = np.diff(b)*y[1:-1]
        area = np.insert(area, 0, d[0]*y[0])
        area = np.append(area,d[-1]*y[-1])
        area = np.absolute(area)
        # cálculo de caudal
        return area

    def plot_compara_historicos(self,**kwargs):
        s = kwargs.get('s',self.aforos()['caudal_medio'])
        filepath = self.folder_path+'historico.png'
        caudal = self.aforo.caudal_medio
        xLabel = r"Caudal$\ [m^{3}/s]$"
        formato = 'png'
        fig = plt.figure(figsize=(14,4.15))
        ax1 = plt.subplot(121)
        ##CUMULATIVE
        ser = s.copy()
        ser.index = range(ser.index.size)
        p25 = s.quantile(0.25)
        p75 = s.quantile(0.75)
        ser.loc[ser.index[-1]+1] = p25
        ser.loc[ser.index[-1]+1] = p75
        ser = ser.sort_values()
        cum_dist = np.linspace(0.,1.,len(ser))
        ser_cdf = pd.Series(cum_dist, index=ser)
        lw=4.0
        ax = ax1.twinx()
        ser_cdf = ser_cdf*100
        ser_cdf.plot(ax=ax,color='orange',drawstyle='steps',label='',lw=lw)
        ser_cdf[ser_cdf<=25].plot(ax = ax,color='g',drawstyle='steps',label='Caudales bajos',lw=lw)
        ser_cdf[(ser_cdf>=25)&(ser_cdf<=75)].plot(ax=ax,color='orange',drawstyle='steps',label='Caudales medios',lw=lw)
        ser_cdf[ser_cdf>=75].plot(ax=ax,color='r',drawstyle='steps',label='Caudales altos',lw=lw)
        #ax.legend(fontsize=14,bbox_to_anchor=(0.5,-0.3),ncol=1)
        #ax.set_title('')
        ax.set_ylabel('Probabilidad [%]',fontsize=16)
        ax.grid()
        ax.set_xlim(0,s.max()*1.05)
        s.hist(ax = ax1,color=self.colores_siata[0],grid=False,bins=20,label='Histograma')
        ax2 = plt.subplot(122)
        ax1.axvline(caudal,color=self.colores_siata[-1],\
                    zorder=40,linestyle='--',\
                    label='Observado',lw=3.0)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, bbox_to_anchor=(1.45,-0.3),ncol=3,fontsize=14)
        ax1.set_ylabel('Frecuencia',fontsize = 16)
        ax1.set_xlabel(xLabel,fontsize = 16)
        for j in ['top','right']:
            ax1.spines[j].set_edgecolor('white')
        for j in ['top','right']:
            ax2.spines[j].set_edgecolor('white')
        ch = s.describe()[1:]
        ch.index = ['Media','Std.','Min.','P25','P50','P75','Max.']
        ch.loc['Obs.'] = caudal
        ch.sort_values().plot(kind = 'barh',ax=ax2,color=tuple(self.colores_siata[-2]),legend=False)
        ax2.yaxis.axes.get_yticklines() + ax2.xaxis.axes.get_xticklines()
        for pos in ['bottom','left']:
            ax2.spines[pos].set_edgecolor(self.colores_siata[-3])
        for pos in ['top', 'right']:
            ax2.spines[pos].set_edgecolor('white')
        size = s.index.size
        #plt.suptitle('Aforos históricos, número de datos = %s'%size,x=0.5,y = 1.05,fontsize=16)
        #ax1.set_title(u'a) Histograma aforos históricos',fontsize = 16)
        ax1.set_title('')
        for p in ax2.patches:
            ax2.annotate('%.2f'%p.get_width(), (p.get_width()*1.04,p.get_y()*1.02),va='bottom',fontsize=16)
        #ax2.set_title(u'b) Resumen aforos históricos',fontsize=16)
        ax2.set_xlabel(xLabel,fontsize = 16)
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        plt.tight_layout()
        text = []
        labels = ax1.get_xticklabels()
        for label in labels:
            text.append(label.get_text())
        ax2.xaxis.set_ticks(map(lambda x:round(x,2),np.linspace(s.min(),s.max(),4)))
        plt.savefig(filepath,format = formato,bbox_inches = 'tight')

    def read_excel_format(self,file):
        df = pd.read_excel(file)
        df = df.loc[df['x'].dropna().index]
        df['vertical'] = range(1,df.index.size+1)
        df['y'] = df['y'].abs()*-1
        df.columns = map(lambda x:x.lower(),df.columns)
        self.seccion = df[self.seccion.columns]
        df = pd.read_excel(file,sheetname=1)
        fecha = pd.to_datetime(df.iloc[1].values[1])
        hora = pd.to_datetime(df.iloc[2].values[1])
        self.aforo.fecha = fecha.strftime('%Y-%m-%d')+hora.strftime(' %H:%M')
        self.aforo['x_sensor'] = df.iloc[4].values[1]
        self.aforo['lamina'] = df.iloc[5].values[1]
        df = pd.read_excel(file,sheetname=2)
        self.levantamiento = df[df.columns[1:]]
        self.levantamiento.columns = ['x','y']
        self.levantamiento.index.name = 'vertical'
        self.aforo.levantamiento = True


    def plot_lluvia_redrio(self,rain,rain_vect,filepath=None):
        fig = plt.figure(figsize=(20,8))
        # lluvia promedio
        ax1 = fig.add_subplot(121)
        ax1.set_ylabel('Intensidad (mm/h)')
        #ax1.set_title('%s - %s'%(rain.argmax(),rain.max()))
        ax1.spines['top'].set_color('w')
        ax1.spines['right'].set_color('w')
        ax1.set_title('Máxima intensidad: {0} - {1}'.format(self.maxint.split(',')[1].split(':')[1],':'.join(self.maxint.split(',')[2].split(':')[1:])))
        rain.plot(ax=ax1,linewidth=1,color='grey') # plot
        ax1.fill_between(rain.index,0,rain.values,facecolor=self.colores_siata[3])
        # lluvia acumulada
        ax2 = fig.add_subplot(122)
        ax2.set_title('Lluvia acumulada')
        self.rain_area_metropol(rain_vect.sum().values/1000.,ax=ax2)
        if filepath:
            plt.savefig(filepath,bbox_inches='tight')

    def plot_bars(self,s,filepath=None,bar_fontsize=14,decimales=2,xfactor =1.005,yfactor=1.01,ax=None):
        if ax is None:
            plt.figure(figsize=(20,6))

        s.plot(kind='bar',ax=ax)
        ax.set_ylim(s.min()*0.01,s.max()*1.01)
        for container in ax.containers:
                  plt.setp(container, width=0.8)
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(),decimales)),
                        (p.get_x() * xfactor, p.get_height() * yfactor),
                        fontsize = bar_fontsize)
        for j in ['top','right']:
            ax.spines[j].set_edgecolor('white')
        ax.set_ylabel(r'$Caudal\ [m^3/s]$')
        if filepath:
            plt.savefig(filepath,bbox_inches='tight')

    def plot_curvas(self,filepath=None):
        fig = plt.figure(figsize=(14,20))
        for i,j in zip(range(1,5),['perimetro','area_total','altura_media','radio_hidraulico']):
            ax = fig.add_subplot(4,2,i)
            ax.scatter(self.aforos.dropna()['caudal_medio'].values,self.aforos.dropna()[j].values)
            ax.scatter(self.aforos.dropna().iloc[-1]['caudal_medio'],self.aforos.dropna().iloc[-1][j])
            ax.set_xlabel('Caudal')
            ax.set_ylabel(j)
        if filepath:
            plt.savefig(filepath,bbox_inches='tight')

    def plot_lluvia(self):
        # entrada
        #paths
        folder_path = pd.to_datetime(self.fecha).strftime('%Y%m%d')
        filepath = self.workspace+folder_path+'/%s'%self.info.slug+'/lluvia.png'
        # dates
        fecha = pd.to_datetime(self.fecha).strftime('%Y%m%d')
        end = pd.to_datetime(fecha)+datetime.timedelta(hours=18)
        start = end - datetime.timedelta(hours=(18+24-6))
        rain = self.radar_rain(start,end)*12.# convert hourly rain (intensity (mm/h))
        rain_vect = self.radar_rain_vect(start,end)
        self.maxint='fecha:%s,maximo:%s mm/h,fecha maximo:%s'%(fecha,rain.max(),rain.argmax())

        if len(rain_vect)>0:
            self.plot_lluvia_redrio(rain,rain_vect,filepath=filepath)
        elif len(rain_vect)==0:
            rain[0]=0.0001
            rain_vect=pd.DataFrame(np.zeros(self.ncells)).T
            self.plot_lluvia_redrio(rain,rain_vect,filepath=filepath)

    def plot_levantamientos(self):
        for id_aforo in self.levantamientos:
            self.plot_section(self.get_levantamiento(id_aforo),x_sensor=2,level=0.0)
            plt.title("%s : %s,%s"%(self.info.slug,self.codigo,id_aforo))

    def procesa_aforo(self):
        self.velocidad_media_dovela()
        self.area_dovela()
        self.seccion['caudal'] = self.seccion.vm*self.seccion.area
        self.perimetro()
        self.aforo.caudal_medio = self.seccion.caudal.sum()
        self.aforo.area_total = self.seccion.area.sum()
        self.aforo.velocidad_media = self.aforo.caudal_medio/self.aforo.area_total
        self.aforo.ancho_superficial = self.seccion['x'].abs().max()-self.seccion['x'].abs().min()
        self.aforo.perimetro = self.seccion.perimetro.sum()
        self.aforo.altura_media = self.seccion['y'].abs()[self.seccion['y'].abs()>0.0].mean()
        self.aforo.radio_hidraulico = self.aforo.area_total/self.aforo.perimetro
        self.fecha = self.aforo.fecha

    def plot_seccion(self):
        self.ajusta_levantamiento()
        self.plot_section(self.levantamiento,xSensor = self.aforo.x_sensor,level=self.aforo.lamina,fontsize=20)
        ax = plt.gca()
        plt.rc('font', **{'size':20})
        ax.scatter(self.aforo.x_sensor,self.aforo.lamina,marker='v',color='k',s=30+30,zorder=22)
        ax.scatter(self.aforo.x_sensor,self.aforo.lamina,color='white',s=120+30+10,edgecolors='k')
        ax.legend()
        ax.set_ylabel('Profundidad [m]')
        ax.spines['top'].set_color('w')
        ax.spines['right'].set_color('w')
        plt.savefig(self.folder_path+'seccion.png',bbox_inches='tight')

    def ajusta_levantamiento(self):
        cond = (self.levantamiento['x']<self.aforo.x_sensor).values
        flag = cond[0]
        for i,j in enumerate(cond):
            if j==flag:
                pass
            else:
                point = (tuple(self.levantamiento.iloc[i-1].values),tuple(self.levantamiento.iloc[i].values))
            flag = j
        intersection = self.line_intersection(point,((self.aforo.x_sensor,0.1*self.levantamiento['y'].min()),(self.aforo.x_sensor,1.1*self.levantamiento['y'].max(),(self.aforo.x_sensor,))))
        self.levantamiento = self.levantamiento.append(pd.DataFrame(np.matrix(intersection),index=['self.aforo.x_sensor'],columns=['x','y'])).sort_values('x')
        self.levantamiento['y'] = self.levantamiento['y']-intersection[1]
        self.levantamiento.index = range(1,self.levantamiento.index.size+1)
        self.levantamiento.index.name = 'vertical'

    def procesa_horarios(self):
        df_alturas = pd.DataFrame(index=self.alturas.index,columns=self.seccion.vertical)
        df_areas = df_alturas.copy()
        df_caudales = df_areas.copy()
        diferencias = self.alturas['profundidad']-self.alturas.loc[pd.to_datetime(self.aforo.fecha).strftime('%H:00'),'profundidad']
        for count,dif in enumerate(diferencias.values):
            alturas = (self.seccion['y'].abs()+dif).values
            alturas[alturas<=0.0] = 0.0
            area = self.get_area(self.seccion['x'].values,alturas)
            caudal = area*self.seccion['vm'].values
            df_alturas.iloc[count] = alturas
            df_areas.iloc[count] = area
            df_caudales.iloc[count] = caudal
        self.h_horaria = df_alturas
        self.a_horaria = df_areas
        self.q_horaria = df_caudales
        self.alturas['caudal'] = self.q_horaria.sum(axis=1).values

    def to_excel(self):
        from pandas import ExcelWriter
        excel_filepath = self.folder_path+'resultado.xlsx'
        writer =  ExcelWriter(excel_filepath)
        informacion = self.aforo.append(self.info_redrio.loc[self.codigo].iloc[:-1].drop('FolderName')).copy()
        try:
            informacion['Subcuenca'] = unicode(informacion['Subcuenca'],errors='ignore')
        except:
            pass
        informacion.to_excel(writer,'informacion',header=False, encoding='utf8')
        workbook  = writer.book
        worksheet = writer.sheets['informacion']
        worksheet.set_column('A:B', 20)
        self.seccion.set_index('vertical').to_excel(writer,'seccion', encoding='utf8')
        self.levantamiento.to_excel(writer,'levantamiento', encoding='utf8')
        self.alturas.index.name = 'Hora'
        self.alturas.fillna('').to_excel(writer,'caudales_horarios', encoding='utf8')
        workbook  = writer.book
        worksheet = writer.sheets['caudales_horarios']
        worksheet.set_column('B:B', 15)
        try:
            self.alturas.to_excel(writer,'profundidades_reportadas')
            self.h_horaria.to_excel(writer,'h_horaria')
            self.a_horaria.to_excel(writer,'a_horaria')
            self.q_horaria.to_excel(writer,'q_horaria')
        except:
            print 'no hourly data'
            pass
        writer.save()

    def get_num_pixels(self,filepath):
        import Image
        width, height = Image.open(open(filepath)).size
        return width,height

    def pixelconverter(self,filepath,width = False,height=False):
        w,h = self.get_num_pixels(filepath)
        factor = float(w)/h
        if width<>False:
            return width/factor
        else:
            return height*factor

    def redrioreport(self,nombre_archivo,nombreEstacion,texto1,texto2,seccion,alturas,lluvia,histograma,resultados,fecha=None,numero_aforos=0,foot=None,head=None,estadisticas=False,heights=True,page2=True,table=True,one_page=False,**kwargs):
        '''
        Generates the reportlab reports of each station included in the attachtments.
        Parameters
        ----------
        nombre_archivo   = path where the pdf report will be generated.
        nombreEstacion   = station name that is going to be used as title of he report.
        texto1           = path to the plain tex file containing the first paragraph of the report which correspond to the descripiton of the registered levels trought the day and the station tranversal section.
        texto2           = path to the plain tex file containing the second paragraph of the report which correspond to the descripiton of the radar antecedent and current rainfall for the campaign date.
        seccion          = path to the png or jpeg file containing a representation of the tranversal section measured.
        alturas          = path to the png or jpeg file containing the hourly level of water registered during the campaign.
        lluvia           = path to the png or jpeg file containing the radar rainfall plots to be analyzed.
        histograma       = path to the png or jpeg file containing the stattistics for the historic gauging campaigns.
        resultados       = path to the excel file containing the gauging campaign data and results for the station.
        fecha (optional) = the gauging campaign date can be set manuall or contained in the results file.
        numero_aforos    = number of gauging campaigns carried out.
        foot (ptional)   = path to the png or jpeg file containing the page foot of the report (Logos)
        head (ptional)   = path to the png or jpeg file containing the page header of the report
        estadisticas(opt)
        heights          = set False to not display the hourly registered levels in the first figure of this report.
        page2            = set False to not display the second page of this report.
        table            = set False to not display the results table.
        one_page         = set True to only display the level and section figure (and its description) and the rainfall figure.
        '''
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate,Paragraph, Table, TableStyle
        from IPython.display import IFrame
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
        import sys

        reload(sys)  # Reload does the trick!
        sys.setdefaultencoding('UTF8')


        barcode_font = r"/media/nicolas/Home/Jupyter/MarioLoco/Tools/AvenirLTStd-Book.ttf"
        pdfmetrics.registerFont(TTFont("AvenirBook", barcode_font))
        barcode_font = r"/media/nicolas/Home/Jupyter/MarioLoco/tools/avenir-next-bold.ttf"
        pdfmetrics.registerFont(TTFont("AvenirBookBold", barcode_font))

        head='/media/nicolas/Home/Jupyter/MarioLoco/tools/head.png' if head==None else head
        foot='/media/nicolas/Home/Jupyter/MarioLoco/tools/foot.png' if foot==None else foot

        print head
        print foot

        texto1=open(texto1).read().decode('utf8')
        texto2=open(texto2).read().decode('utf8')

        resultados=self.aforo
        fecha=self.aforo.fecha

        try:
            dispositivo=resultados.loc['dispositivo'].values[0]
        except:
            dispositivo='OTT MF-PRO'


        textf1 = kwargs.get('textf1','Figura 1. a) Dibujo de la sección transversal del canal. b) Caudales horarios obtenidos a partir de profundidades de la lámina de agua.')
        textf2 = 'Tabla 1. Resumen, muestra el dispositivo con el que se realizó el aforo y los parámetros hidráulicos estimados más relevantes.'
        textf3 = kwargs.get('textf3','Figura 2. a) Distribución temporal de la lluvia en la cuenca. La sombra azul invertida representa la intensidad promedio en mm/h. b) Distribución espacial de la lluvia acumulada en la cuenca en mm en un periodo de 36 horas.')
        text_color = '#%02x%02x%02x' % (8,31,45)
        widthPage =  816
        heightPage = 1056
        pdf = canvas.Canvas(nombre_archivo,pagesize=(widthPage,heightPage))
        cx = 0
        cy = 900
        #pdf.drawImage(ruteSave,20,250,width=860,height=650)
        pdf.drawImage(foot,816/2-(100/(209/906.))/2,10,width=(100/(209/906.)),height=100)
        pdf.drawImage(head,0,1056-129,width=816,height=129)
        text_color = '#%02x%02x%02x' % (8,31,45)
        styles=getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Texts',\
                                  alignment=TA_CENTER,\
                                  fontName = "AvenirBook",\
                                  fontSize = 20,\
                                  textColor = text_color,\
                                  leading = 20))

        styles.add(ParagraphStyle(name='Justify',\
                                  alignment=TA_JUSTIFY,\
                                  fontName = "AvenirBook",\
                                  fontSize = 14,\
                                  textColor = text_color,\
                                  leading = 20))

        styles.add(ParagraphStyle(name='JustifyBold',\
                              alignment=TA_JUSTIFY,\
                              fontName = "AvenirBookBold",\
                              fontSize = 13,\
                              textColor = text_color,\
                              leading = 20))
        #flagheigths
        if heights == False:
            height = 180
            width = self.pixelconverter(seccion,height=height)
            xloc = widthPage/2.0 - (width/2.0)
            pdf.drawImage(seccion,xloc,550,width = width,height = height)
            p = Paragraph('Figura 1. Dibujo de la sección transversal del canal', styles["JustifyBold"])
            p.wrapOn(pdf, 716, 200)
            p.drawOn(pdf,270,490)

        else:
            pdf.drawImage(seccion,50,550,width=310,height=211)
            pdf.setFont("AvenirBook", 14)
            pdf.drawString(220,770,"a)")
            pdf.drawString(600,770,"b)")
            pdf.drawImage(alturas,50+310,550,width=426-13,height=211)


            p = Paragraph(textf1, styles["JustifyBold"])
            p.wrapOn(pdf, 716, 200)
            p.drawOn(pdf,50,480)

        if len(texto1)<470:
            p = Paragraph(texto1, styles["Justify"])
            p.wrapOn(pdf, 720, 200)
            p.drawOn(pdf,50,830)

        elif len(texto1)<540:
            p = Paragraph(texto1, styles["Justify"])
            p.wrapOn(pdf, 720, 200)
            p.drawOn(pdf,50,815)

        elif len(texto1)<580:
            p = Paragraph(texto1, styles["Justify"])
            p.wrapOn(pdf, 720, 200)
            p.drawOn(pdf,50,800)

        elif len(texto1)<640:
            p = Paragraph(texto1, styles["Justify"])
            p.wrapOn(pdf, 720, 200)
            p.drawOn(pdf,50,775)

        elif len(texto1)<700:
            p = Paragraph(texto1, styles["Justify"])
            p.wrapOn(pdf, 720, 200)
            p.drawOn(pdf,50,760)

        else:
            p = Paragraph(texto1, styles["Justify"])
            p.wrapOn(pdf, 720, 200)
            p.drawOn(pdf,50,750)


        pdf.setFillColor(text_color)
        pdf.setFont("AvenirBook", 20)
        print nombreEstacion

        p = Paragraph(u'Estación %s - %s'%(nombreEstacion.encode('utf8'),fecha), styles["Texts"])
        p.wrapOn(pdf, 816, 200)
        p.drawOn(pdf,0,945)

        data= [['Caudal total [m^3/s] ', round(float(resultados.caudal_medio),2), 'Dispositivo', dispositivo],
               [u'Área mojada [m^2]',round(float(resultados.area_total),2), 'Ancho superficial [m]',round(float(resultados.ancho_superficial),2)],
               ['Profundidad media [m]', round(float(resultados.altura_media),2), 'Velocidad promedio [m/s]',round(float(resultados.velocidad_media),2)],
               [u'Perímetro mojado [m]', round(float(resultados.perimetro),2), 'Radio hidráulico [m]', round(float(resultados.radio_hidraulico),2)],]

        if table==True:
            t=Table(data,colWidths = [210,110,210,110],rowHeights=[30,30,30,30],style=[('GRID',(0,0),(-1,-1),1,text_color),
                                ('ALIGN',(0,0),(0,-1),'LEFT'),
                                ('BACKGROUND',(0,0),(0,-1),colors.white),
                                ('ALIGN',(3,2),(3,2),'LEFT'),
                                ('BOX',(0,0),(-1,-1),1,colors.black),
                                ('TEXTFONT', (0, 0), (-1, 1), 'AvenirBook'),
                                ('TEXTCOLOR',(0,0),(-1,-1),text_color),
                                ('FONTSIZE',(0,0),(-1,-1),14),
                                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                ('ALIGN',(1,0),(1,-1),'CENTER'),
                                ('ALIGN',(3,0),(3,-1),'CENTER')
            ])

            t.wrapOn(pdf, 650, 200)
            t.drawOn(pdf,100,310)


            p = Paragraph(textf2, styles["JustifyBold"])
            p.wrapOn(pdf, 716, 200)
            p.drawOn(pdf,50,240)

        pdf.setFont("AvenirBookBold", 14)
        pdf.setFillColor('#%02x%02x%02x' % (8,31,45))
        pdf.setFont("AvenirBook", 15)
        pdf.setFillColor('#%02x%02x%02x' % (8,31,45))


        if one_page==True:
            page2=False
            height = 225
            width = self.pixelconverter(lluvia,height=height)
            xloc = widthPage/2.0 - (width/2.0)
            pdf.drawImage(lluvia,xloc,230,width = width,height = height)
            pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/acumuladoLegend.jpg',642,255,width=43.64,height=200)
            p = Paragraph(textf3, styles["JustifyBold"])
            p.wrapOn(pdf, 716, 200)
            p.drawOn(pdf,50,130)

        pdf.showPage()

        #PÁGINA 2

        if page2==True:
            pdf.drawImage(foot,816/2-(100/(209/906.))/2,10,width=(100/(209/906.)),height=100)
            pdf.drawImage(head,0,1056-129,width=816,height=129)
            height = 225
            width = self.pixelconverter(lluvia,height=height)
            xloc = widthPage/2.0 - (width/2.0)
            pdf.drawImage(lluvia,xloc,540,width = width,height = height)

	    if len(texto2)<500:
            	p = Paragraph(texto2, styles["Justify"])
            	p.wrapOn(pdf, 720, 200)
            	p.drawOn(pdf,50,820)

	    else:
                p = Paragraph(texto2, styles["Justify"])
                p.wrapOn(pdf, 720, 200)
                p.drawOn(pdf,50,790)

            textf4 = 'Figura 3. a) Distribuciones de frecuencia, número de aforos: %s, la línea punteada vertical es el caudal observado, la curva es una distribución de frecuencia acumulada que presenta el régimen de caudales. b) Resumen de estadísticos. Max = Caudal máximo, Min = Caudal mínimo, P25 = Percentil 25, P50 = Mediana, P75 = Percentil 75, Media = Caudal promedio, Std = desviación estándar, Obs = Caudal observado.'%(numero_aforos)
            p = Paragraph(textf3, styles["JustifyBold"])
            p.wrapOn(pdf, 716, 200)
            p.drawOn(pdf,50,480)

            # distribuciones
            if numero_aforos>0:
                if estadisticas == False:
                    height = 230
                    width = self.pixelconverter(histograma,height=height)
                    xloc = widthPage/2.0 - (width/2.0)
                    pdf.drawImage(histograma,xloc,220,width = width,height = height)
                    p = Paragraph(textf4, styles["JustifyBold"])
                    p.wrapOn(pdf, 716, 200)
                    p.drawOn(pdf,50,125)
                    pdf.setFont("AvenirBook", 14)
                    pdf.drawString(205,460,"a)")
                    pdf.drawString(590,460,"b)")

                else:
                    textf4 = estadisticas
                    pdf.drawImage(histograma,155,180,width=500,height=250)
                    p = Paragraph(textf4, styles["JustifyBold"])
                    p.wrapOn(pdf, 716, 200)
                    p.drawOn(pdf,120,145)

                p = Paragraph(u'Estación %s - %s'%(nombreEstacion.encode('utf8'),fecha), styles["Texts"])
                p.wrapOn(pdf, 816, 200)
                p.drawOn(pdf,0,945)
                # pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/acumuladoLegend.jpg',642,570,width=43.64,height=200)
                pdf.drawImage('/media/nicolas/Home/Jupyter/MarioLoco/tools/arrow.png',kwargs.get('left',595),575,width=20,height=20)
                left=kwargs.get('left',590)
                pdf.drawString(left+10,596,"N")
                pdf.setFont("AvenirBook", 14)
                pdf.setFillColor('#%02x%02x%02x' % (8,31,45))
                if left>560:
                    pdf.drawString(205,770,"a)")
                    pdf.drawString(590,770,"b)")
                x = 460
            else:
                p = Paragraph(u'Estación %s - %s'%(nombreEstacion.encode('utf8'),fecha), styles["Texts"])
                p.wrapOn(pdf, 816, 200)
                p.drawOn(pdf,0,945)
        else:
            1

        pdf.save()
