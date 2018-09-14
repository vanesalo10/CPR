# CPR
Consultas Plots y Reportes (CPR): Repositorio con funciones y clases para la realización de: Consultas, plots y reportes de las estaciones y los modelos que administra hidrología

**CPR** contiene funciones y clases destinadas a facilitar el trabajo con los datos de SIATA.
Este modulo divide su forma de operar de la siguiente manera:

- Clases de consulta de información de estaciones.
- Realización de plots genéricos de estaciones y resultados de modelos.
- Generación de reportes.
- contiene scripts que pueden ser ejecutados desde **bash**.

## Requisitos:

Para su correcta ejecución se deben tener los siguientes requisitos:

- Estar al interior de la red de SIATA para la realizaciónd e consultas.
- Para consultar y transformar radar requiere:
	- Conexión a los datos de radar.
	- El módulo de Radar: https://github.com/nicolas998/Radar.git
- Los siguientes paquetes de **Python**:
	- Pandas.
	- numpy.
	- matplotlib.
	- datetime.
	- reportlab.


## Instalación:

Para instalar este paquete en su equipo debe en una terminal de bash indicar el siguiente
codigo:

Primero debe desplazar la **terminal** hasta la carpeta *clonada* o *descargada* de este repo.

```bash
cd Path/CPR
```

Luego se realiza la instalación, para esto debe tener privilegios de **sudoer**.

```bash
sudo python setup.py install
```

## Contacto:

Dudas sobre el codigo, y sugerencias:

- Para dudas escribir a: *hidrosiata@gmail.com* o a *mario.cano@siata.gov.co*
- Reporte de bugs y problemas favor escribirlos en: https://github.com/SIATAhidro/CPR/issues

## Instalar las siguientes dependencias para instalar bases de datos.
sudo pip install mysql-connector-python
sudo pip instal mysqlclient
sudo pip install SQLAlchemy
sudo pip install reportlab

#Instalar WMF.
# Configurar idioma
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
# al ejecutar la sgte. linea responder Y a lo que aparece.
sudo dpkg-reconfigure locales



Modificar archivo cpr/info.py, modificar base de datos local para cambiar las credenciales. cambiar radar_path y data_path por la ruta donde se encuentra la carpeta de datos.
