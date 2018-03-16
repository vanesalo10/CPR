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
#PYTHON CONFIGURATION
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
# configure locale for data in spanish
#sudo locale-gen es_ES.UTF-8
#sudo dpkg-reconfigure locales
# Siata settings
plt.rc('font', family=fm.FontProperties(fname='/media/nicolas/maso/Mario/tools/AvenirLTStd-Book.ttf',).get_name())
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
        conn_db = MySQLdb.connect(self.host,self.user,self.passwd,self.dbname)
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
        df = pd.read_sql(sql,self.conn_db,*keys,**kwargs)
        if close_db == True:
            self.conn_db.close()
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
    # functions for id_hydro
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

    def fecha_hora_format_data(self,field,start,end):
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
        df = self.read_sql("SELECT fecha,hora,%s from datos WHERE cliente = '%s' and calidad = '1' and %s"%format)
        # converts centiseconds in 0
        try:
            df['hora'] = df['hora'].apply(lambda x:x[:-3]+':00')
        except TypeError:
            df['hora'] = df['hora'].apply(lambda x:str(x)[-8:])
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
            print("progress: %.1f %% - %s out of %s"%format)
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
        start,end = pd.to_datetime(start),pd.to_datetime(end)
        files = os.listdir(self.rain_path)
        if files:
            for file in files:
                comienza,finaliza,codigo,usuario = self.file_format_to_variables(file)
                if (comienza<=start) and (finaliza>=end) and (codigo==self.codigo):
                    file =  file[:file.find('.')]
                    break
                else:
                    file = None
        else:
            file = None
        return file

    def radar_rain(self,start,end,ext='.hdr'):
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
	    save =  '%s%s'%(self.rain_path,self.file_format(start,end))
            self.get_radar_rain(start,end,self.info.nc_path,self.radar_path,save,converter=converter,utc=True)
            print file
            file = self.rain_path + self.check_rain_files(start,end)
            if ext == '.hdr':
                obj =  self.hdr_to_series(file+'.hdr')
            else:
                obj =  self.bin_to_df(file)
            obj = obj.loc[start:end]
        return obj

    def radar_rain_vect(self,start,end):
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
        return self.radar_rain(start,end,ext='.bin')

    def sensor(self,start,end):
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
        s = sql.fecha_hora_format_data(['pr','NI'][self.info.tipo_sensor],start,end)
        return s
        
    def level(self,start,end,offset='new'):
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
        s = self.sensor(start,end)
        if offset == 'new':
            return self.info.offset - s
        else:
            return self.info.offset_old - s

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

    def plot_basin_rain(self,vec,ax=None):
        if ax is None:
            fig = plt.figure(figsize=(10,16))
            ax = fig.add_subplot()
        cmap_radar,levels,norm = self.radar_cmap()
        extra_lat,extra_long = self.adjust_basin(fac=0.02)
        mapa,contour = self.basin_mappable(vec,
                                      ax=ax,
                                      extra_long=extra_long,
                                      extra_lat = extra_lat,
                                      contour_keys={'cmap'  :cmap_radar,
                                                    'levels':levels,
                                                    'norm'  :norm},
                                     perimeter_keys={'color':'red'})

        cbar = mapa.colorbar(contour,location='bottom',pad="5%")
        mapa.readshapefile(self.info.net_path,'net_path')
        mapa.readshapefile(self.info.net_path,'stream_path')
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
    
    def plot_operacional(self,series,window,filepath):
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
        if window == '3h':
            lamina = 'current'
        else:
            lamina = 'max'
        try:
            self.plot_level(series,
                            lamina=lamina,
                            risk_levels=np.array(self.risk_levels)/100.0,
                            resolution='m',
                            ax=ax,
                            scatter_size=40)
        
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
        bat = self.last_bat(self.info.x_sensor)
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
            text= u'Estación de Nivel tipo %s\nResolución temporal: 1 minutos\n%% de datos transmitidos: %.2f\nProfundidad máxima: %.2f [m]\nNivel de riesgo máximo: %s\nProfundidad promedio: %.2f [m]\n*Calidad de datos a\xfan\n sin verificar exhaustivamente'%(format)
        except:
            text = u'ESTACIÓN SIN DATOS TEMPORALMENTE'
        ax3 = fig.add_subplot(313)
        ax3.text(0.0,1.3,'RESUMEN',color = self.colores_siata[-1])
        ax3.text(0.0, 0.0,text,linespacing=2.1)
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
            obj = Nivel(codigo = codigo,user='sample_user',passwd='s@mple_p@ss',SimuBasin=False)
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
        for horas,valor in zip([1,3,24,72],[1,2,3,4]):
            r = s['fecha']<(now-datetime.timedelta(hours=horas))
            s.loc[s[r].index,'rango']=valor
        return s.dropna()
