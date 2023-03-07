"""
Retrieving weather data from CSW.

"""
from datetime import datetime
import numpy as np

class BayWeather:
    """Class to get weather data from CSW.

    Attributes
    ----------
    lon
        longitude decimal degree coordinate.
    lat
        latitude decimal degree coordinate.
    start
        start of the weather series.
    end     
        end of the weather series.
        
    Methods
    -------
    TWC()
        get weather data from The Weather Company (TWC) data stored in CSW.
    WPD()
        compute Wetting Period Duration (WPD), a important weather feature for crop protection.
    """

    def __init__(self, lon: float, lat: float, start: datetime, end: datetime) -> None:
        self.lon = lon
        self.lat = lat
        self.start = start.replace(tzinfo=None)
        self.end = end.replace(tzinfo=None)

    def TWC(self) -> pd.DataFrame:
        """Method to get daily weather series from The Weather Company (TWC) source in CSW.

        Attributes
        ----------
        None

        Returns
        -------
        TWC()
            dataframe obtained from TWC-CSW.

        """
        rad = np.pi / 180  # Radians to degrees
        gra = 180 / np.pi  # Degrees to radians

        query_wth = """ 
            DECLARE locations ARRAY<STRUCT<u_lat FLOAT64, u_lon FLOAT64>> DEFAULT [({},{})];
            DECLARE u_start_date DATE DEFAULT DATE('{}');
            DECLARE u_end_date DATE DEFAULT DATE('{}');
            DECLARE u_variables STRING DEFAULT 'min_temperature,max_temperature,total_precipitation,avg_relative_humidity,avg_wind_speed,total_net_solar_radiation';
            DECLARE uom STRING DEFAULT 'm';
            CALL `historical_weather.historical_weather_daily_blend`(locations, u_start_date, u_end_date, u_variables, uom);
            """.format(
            self.lat, self.lon, self.start, self.end
        )

        df_wth = (
            CSWconnect("location360-datasets").load(query_wth).sort_values(by="date")
        )
        df_wth["DOY"] = df_wth["date"].apply(lambda x: int(x.strftime("%j")))

        ## Incident solar radiation (Rs_in)
        Ko = 37.63 * (1 + (0.033 * (np.cos(rad * ((360 * df_wth["DOY"]) / 365)))))
        ds = 23.45 * np.sin(rad * (360 * (df_wth["DOY"] - 80) / 365))
        hn = (np.arccos(-np.tan(rad * df_wth["lat"][0]) * np.tan(rad * ds))) * gra
        Qo = Ko * (
            rad * hn * np.sin(rad * df_wth["lat"][0]) * np.sin(rad * ds)
            + np.cos(rad * df_wth["lat"][0]) * np.cos(rad * ds) * np.sin(rad * hn)
        )

        df_wth["downward_solar_radiation"] = (
            0.16
            * Qo
            * ((df_wth["max_temperature"]) - (df_wth["min_temperature"])) ** 0.5
        )
        
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
              elevation
            FROM 
              twc_cod, point 
            WHERE 
              geohash4 IN (ST_GEOHASH(point.pt_geom, 4))
            ORDER BY 
              ST_DISTANCE(point.pt_geom, twc_cod.geom) LIMIT 1
              """.format(lon, lat)

        elev = CSWconnect("location360-datasets").load(query_elev).squeeze()
       
        df_wth['elevation'] = elev

        return df_wth

    def WPD(self) -> pd.DataFrame:
        """Method to get hourly relative humidity data from TWC-CSW and to compute WPD. 

        Attributes
        ----------
        None

        Returns
        -------
        WPD()
             dataframe representing the number of night hours that relative humidity was higher than 90%.

        """
        query_wth = """ 
            DECLARE locations ARRAY<STRUCT<u_lat FLOAT64, u_lon FLOAT64>> DEFAULT [({},{})];
            DECLARE u_start_date DATE DEFAULT DATE('{}');
            DECLARE u_end_date DATE DEFAULT DATE('{}');
            DECLARE u_variables STRING DEFAULT 'relative_humidity';
            DECLARE uom STRING DEFAULT 'm';
            CALL `historical_weather.historical_weather_hourly_blend`(locations, u_start_date, u_end_date, u_variables, uom);
            """.format(
            self.lat, self.lon, self.start, self.end
        )

        df_rh = (
            CSWconnect("location360-datasets").load(query_wth).sort_values(by="local_time")
        )

        df_rh.index = pd.to_datetime(df_rh["local_time"], format="%Y-%m-%d %H:%M:%S")
        df_rh.sort_index(inplace=True)

        df_rh_night = df_rh["relative_humidity"].between_time("18:00", "06:00")

        df_rh_90 = df_rh_night[df_rh_night > 90]

        WPD = df_rh_90.groupby(
            by=[df_rh_90.index.year, df_rh_90.index.month, df_rh_90.index.day]
        ).count()
               
        
        WPD.index = pd.to_datetime(
            WPD.index.get_level_values(0).astype(str)
            + "-"
            + WPD.index.get_level_values(1).astype(str)
            + "-"
            + WPD.index.get_level_values(2).astype(str),
            format="%Y-%m-%d",
        )
        
        WPD.name = 'WPD'

        return WPD
