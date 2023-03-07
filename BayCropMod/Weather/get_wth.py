class DSSATwth:
    """Class to get weather data prepared to DSSAT simulations.

    Attributes
    ----------
    credentials  
        credentials to access Google BigQuery.
    project  
        name of the project which contain the datasets (e.g., 'product360-datasets').

    Methods
    -------
    load()
        Get pandas dataframe from query used.
    save()
        Save pandas dataframe as a table in CSW.
    """

    def __init__(self, lon: float, lat: float, start: ) -> None:
        self.credentials = credentials
        self.project = project
        self.bq_client = bigquery.Client(
                project=project, credentials=credentials)

        # Weather retrieval
        project_loc = 'location360-datasets'

        query_wth = """ 
            DECLARE locations ARRAY<STRUCT<u_lat FLOAT64, u_lon FLOAT64>> DEFAULT [({},{})];
            DECLARE u_start_date DATE DEFAULT DATE('{}');
            DECLARE u_end_date DATE DEFAULT DATE('{}');
            DECLARE u_variables STRING DEFAULT 'min_temperature,max_temperature,total_precipitation,avg_relative_humidity';
            DECLARE uom STRING DEFAULT 'm';
            CALL `historical_weather.historical_weather_daily_blend`(locations, u_start_date, u_end_date, u_variables, uom);
            """.format(lat, lon, st, (hd + timedelta(days=120)).date())

        df_wth = CSWconnect(project_loc).load(query_wth).sort_values(by='date')
        df_wth['DOY'] = df_wth['date'].apply(lambda x: int(x.strftime('%j')))

        ## Incident solar radiation (Rs_in)
        rad = np.pi/180 # Radians to degrees
        gra = 180/np.pi # Degrees to radians

        Ko = 37.63*(1+(0.033*(np.cos(rad*((360*df_wth['DOY'])/365)))))
        ds = 23.45*np.sin(rad*(360*(df_wth['DOY']-80)/365))
        hn = (np.arccos(-np.tan(rad*df_wth['lat'][0])*np.tan(rad*ds)))*gra
        Qo = Ko*(rad*hn*np.sin(rad*df_wth['lat'][0])*np.sin(rad*ds)+
                np.cos(rad*df_wth['lat'][0])*np.cos(rad*ds)*np.sin(rad*hn))

        df_wth['Rs_in'] = 0.16*Qo*((df_wth['max_temperature'])-(df_wth['min_temperature']))**0.5


        WTH_columns = [
                'date',
                'min_temperature',
                'max_temperature',
                'total_precipitation',
                'Rs_in',
                'avg_relative_humidity',
                ]
        # Create a WeatherData instance
        WTH_DATA = WeatherData(df_wth.loc[:, WTH_columns],
                variables={
                    'min_temperature': 'TMIN',
                    'max_temperature': 'TMAX',
                    'total_precipitation': 'RAIN',
                    'Rs_in': 'SRAD',
                    'avg_relative_humidity': 'RHUM',
                    })

                query_elev = """ 
            WITH twc_cod AS (
              SELECT 
                grid_id, elevation, lat, lon, geohash4, geom 
              FROM 
                historical_weather.twc_cod_grids 
              ),
              point AS (
              SELECT 
                ST_GEOGPOINT({},{}) as pt_geom 
              )
            SELECT 
              elevation, lat, lon 
            FROM 
              twc_cod, point 
            WHERE 
              geohash4 IN (ST_GEOHASH(point.pt_geom, 4))
            ORDER BY 
              ST_DISTANCE(point.pt_geom, twc_cod.geom) LIMIT 1
              """.format(lon, lat)

        elev = CSWconnect(project_loc).load(query_elev)

        # Create a WheaterStation instance
        wth = WeatherStation(WTH_DATA, {
            'ELEV': elev['elevation'][0],
            'LAT': lat,
            'LON': lon,
            'INSI': 'dpoes'
            })
