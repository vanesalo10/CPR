import cprv1 as cpr

class Humedad(cpr.SqlDb):
    '''
    Provide functions to manipulate data related
    to soil moisture sensors.
    '''
    local_table  = 'estaciones_estaciones'
    remote_table = 'estaciones'
    
    def __init__(self,user,passwd,codigo = None,**kwargs):
        '''
        The instance inherits modules to manipulate SQL
        Parameters
        ----------
        codigo        : primary key
        remote_server :
        local_server  : database kwargs to pass into the Sqldb class
        nc_path       : path of the .nc file to set wmf class
        '''
        self.data_path ='/media/nicolas/maso/Mario/'
        cpr.SqlDb.__init__(self,codigo=codigo,user=user,passwd=passwd,table=None,**cpr.info.LOCAL)
        
        self.colores_siata = [[0.69,0.87,0.93],[0.61,0.82,0.88],[0.32,0.71,0.77],[0.21,0.60,0.65],#\
                                  [0.0156,0.486,0.556],[0.007,0.32,0.36],[0.0078,0.227,0.26]]
    @property
    def info(self):
        query = "SELECT * FROM %s WHERE clase = 'Humedad' and codigo='%s'"%(self.local_table,self.codigo)
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
        query = "SELECT * FROM %s WHERE clase ='humedad'"%(self.local_table)
        return self.read_sql(query).set_index('codigo')
    
    def read_humedad(self,start,end):        
        s = cpr.Nivel(**cpr.info.REMOTE).read_sql("select fecha_hora, h1, h2, h3, c1, c2, c3, t1, t2, t3, vw1, vw2, vw3 from humedad_rasp where cliente = '%s' and fecha_hora between '%s' and '%s'"%(self.codigo,start,end)).set_index('fecha_hora')
        s = s.loc[s.index.dropna()]
        s[s<0.0] = np.NaN
        return s