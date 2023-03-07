#%% [markdown]
# # Recommendation system
# ## Imports


#%%
#data extraction
import hvac
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas_gbq

#plot and handle
import os
import json
from kedro.extras.datasets.pandas import GBQQueryDataSet
import pandas as pd
import geopandas as gpd
from shapely import wkt
from datetime import datetime, timedelta
#import folium
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calplot
import pickle

#computing
from tqdm.auto import tqdm
from p_tqdm import p_map

#display
import sweetviz as sv
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"/mnt/imported/code/Analysis_scripts/DS/learntorank/conf/base/creds.json"


# %%
os.environ['DOMINO_USER_API_KEY']

#%% [markdown]
# ## Helper Functions

#%% [markdown]
# ## Data Collection
# # MD database

#%%

stab_sql = """SELECT YEAR, Head, Other, Field_name, Head_mean, Other_mean, EI
              FROM latam_datasets.stab_brazil_corn"""

df = GBQQueryDataSet(stab_sql, project='bcs-market-dev-lake').load()
df.head()

#%%
df.info()

#%% [markdown]
# # Location 360 database
# # Weather 

# %%


#%% [markdown]
# ## Coordinates as input
coords = {'long':-45.4692, 'lat':-12.8623}

# %%
grid_weather = """WITH twc_cod AS (
  SELECT grid_id, elevation, geohash4, geom as twc_geom 
  FROM location360-datasets.historical_weather.twc_cod_grids 
  ),
  point AS (
    SELECT ST_GEOGPOINT({}, {}) as p_geom 
  )
    SELECT grid_id, elevation, twc_geom FROM twc_cod, point WHERE geohash4 = ST_GEOHASH(point.p_geom, 4) 
    ORDER BY ST_DISTANCE(point.p_geom, twc_cod.twc_geom)
    LIMIT 5""".format(coords['long'], coords['lat'])

loc = GBQQueryDataSet(grid_weather, project='location360-datasets').load()
loc

# %%
loc['twc_geom'] = loc['twc_geom'].apply(wkt.loads)
loc_gdf = gpd.GeoDataFrame(loc, geometry='twc_geom')

# %% [markdown]
# ## Mapping 
map = folium.Map(location=[coords['lat'], coords['long']], zoom_start=12)
folium.Marker(
    location=[coords['lat'], coords['long']],
    popup="Location",
    icon=folium.Icon(color="green"),
).add_to(map)

loc_fol = folium.features.GeoJson(loc_gdf.to_json())
map.add_children(loc_fol)
map

# %% [markdown]
# ## Nearest grid
near_ppt=loc_gdf.loc[[0]]
near_ppt['grid_id'].squeeze()

# %% [markdown]
# ## Precipitation time series
ppt_ts_sql = f"""SELECT * 
  FROM location360-datasets.historical_weather.twc_high_resolution_precipitation_metric_daily 
  WHERE grid_id = '{near_ppt['grid_id'].squeeze()}' 
  ORDER BY date"""

ppt_ts = GBQQueryDataSet(ppt_ts_sql, project='location360-datasets').load()


# %%
ppt_ts

# %%[markdown]
# # Weather blend Location360

# SET all_variables = ['grid_id','lat','lon','date',
# 'min_temperature','max_temperature','max_dew_point_temperature',
# 'min_dew_point_temperature','avg_dew_point_temperature','max_wind_speed',
# 'min_wind_speed','avg_wind_speed','max_wind_gust','avg_wind_direction',
# 'max_relative_humidity','min_relative_humidity','avg_relative_humidity',
# 'max_atmospheric_pressure','min_atmospheric_pressure','avg_atmospheric_pressure',
# 'max_precipitation_rate','total_precipitation','total_downward_solar_radiation',
# 'max_downward_solar_radiation','total_net_solar_radiation','max_net_solar_radiation',
# 'avg_total_cloud_cover','avg_snow_depth','avg_snow_density','max_soil_temperature_level_1',
# 'max_soil_temperature_level_2','max_soil_temperature_level_3','max_soil_temperature_level_4',
# 'min_soil_temperature_level_1','min_soil_temperature_level_2','min_soil_temperature_level_3',
# 'min_soil_temperature_level_4','avg_soil_temperature_level_1','avg_soil_temperature_level_2',
# 'avg_soil_temperature_level_3','avg_soil_temperature_level_4','avg_soil_moisture_level_1',
# 'avg_soil_moisture_level_2','avg_soil_moisture_level_3','avg_soil_moisture_level_4', 'day_length'];


# %%




# %%
from datetime import datetime

num_cores = multiprocessing.cpu_count()


yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')


seq_date=pd.date_range(start="1999-12-31",end=yesterday, periods=num_cores).to_pydatetime()
seq_date
# %%
seq_date[1].date() + timedelta(days=1)
# %%




# %%
queries[0:3]

# %%
def runQuery(query): 
  df=GBQQueryDataSet(query, project='location360-datasets').load()
  return df
# %%
result = map(lambda x: GBQQueryDataSet(x, project='location360-datasets').load(), queries[0:3])


# %%

# %%
teste.plot()
# %%

pool = multiprocessing.Pool(num_cores)

results = pool.map(runQuery, queries[0:3])

# %%
import pandas as pd
import calplot
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import os
from kedro.extras.datasets.pandas import GBQQueryDataSet
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"/mnt/imported/code/Analysis_scripts/DS/learntorank/conf/base/creds.json"

from tqdm.auto import tqdm
from p_tqdm import p_map

# %%

df_raw = pd.read_csv('../data/01_raw/protocols_cp.csv')
df_raw
# %%

df = df_raw[['FIELD_Loc_longitude','FIELD_Loc_latitude','FIELD_locationName','Clima','FIELD_plantingDate','FIELD_harvestDate']]
df.columns = ['lon', 'lat','loc','weather','seed','harvs']

## Filter to choosed areas by Ximena
df = df.loc[[0,1,4,5,6,7,9,11]]
df[['seed','harvs']] = df[['seed','harvs']].apply(pd.to_datetime, format="%Y-%m-%d")

# %%
#df=df.loc[[0]]
df

# %%
def runQuery(query): 
      df=GBQQueryDataSet(query, project='location360-datasets').load()
      return df

#num_cores = os.cpu_count()-2
num_cores = 5
yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
seq_date=pd.date_range(start="2021-12-31",end=yesterday, periods=num_cores).to_pydatetime()


# %%

## Wetting-Period Duration Heat Map

for index, row in df.iterrows():

  t = time.time()
  coords = {'lon':row['lon'], 'lat':row['lat']}
  
  queries = [] 

  for i in range(num_cores - 1): 
    query_i = """ 
    DECLARE locations ARRAY<STRUCT<u_lat FLOAT64, u_lon FLOAT64>> DEFAULT [({},{})];
    DECLARE u_start_date DATE DEFAULT DATE('{}');
    DECLARE u_end_date DATE DEFAULT DATE('{}');
    DECLARE u_variables STRING DEFAULT 'relative_humidity';
    DECLARE uom STRING DEFAULT 'm';
    CALL `location360-datasets.historical_weather.historical_weather_hourly_blend`(locations, u_start_date, u_end_date, u_variables, uom);
    """.format(coords['lat'], coords['lon'], seq_date[i].date() + timedelta(days=1), seq_date[i + 1].date()) 
    queries.append(query_i)

    
  results = p_map(runQuery, queries, **{"num_cpus": num_cores})

  df_rh = pd.concat(results)
  df_rh.index = pd.to_datetime(df_rh['local_time'],format='%Y-%m-%d %H:%M:%S')
  df_rh.sort_index(inplace=True)

  df_rh_night = df_rh['relative_humidity'].between_time('18:00', '06:00')
    
  df_rh_90 = df_rh_night[df_rh_night > 90]
    
  WPD = df_rh_90.groupby(by=[df_rh_90.index.year, df_rh_90.index.month, df_rh_90.index.day]).count()
  WPD.index = pd.to_datetime(WPD.index.get_level_values(0).astype(str) + '-' +
              WPD.index.get_level_values(1).astype(str) + '-' +
              WPD.index.get_level_values(2).astype(str),
              format='%Y-%m-%d')

  WPD_hours = 10   
  WPD_count = WPD.groupby(by=[WPD.index.year, WPD.index.month]).apply(lambda x: len(x[x>=WPD_hours]))

  WPD_count.index = pd.to_datetime(WPD_count.index.get_level_values(0).astype(str) + '-' +
              WPD_count.index.get_level_values(1).astype(str),
              format='%Y-%m')
  
  
  
  WPD_count.plot.line()
  plt.xlabel('Time')
  plt.ylabel('Days with WPD >= 10h')
  plt.title(row['loc'])
  plt.savefig('../data/01_raw/WPD_month' + row['loc'] + '.png', bbox_inches = 'tight')

# %%
  WPD_count.plot.line()
  plt.xlabel('Time')
  plt.ylabel('Days with WPD >= 10h')
  plt.title(row['loc'])
  plt.savefig('../data/01_raw/WPD_month_' + row['loc'] + '.png', bbox_inches = 'tight')


# %%

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(WPD_count.index, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')

plt.show()



  # %%
  
  start_day = pd.to_datetime('2017-01-01')
  end_day = pd.to_datetime(yesterday)

  WPD_serie = WPD.loc[start_day:end_day] 

  suptitle_kws = dict(ha='center',size=22)
  pl1 = calplot.calplot(WPD_serie,cmap = 'RdYlGn_r', figsize = (16, 14), 
                        textformat  ='{:.0f}',  suptitle_kws = suptitle_kws,
                        edgecolor='black', linewidth=2,
                        suptitle = row['loc'] + ", weather " + row['weather'] +" \n Number of hours at night with RH > 90% (Wetting-Period Duration - WPD)")

  plt.savefig('../data/02_intermediate/WPD_hist_' + row['loc'] + '.png', bbox_inches = 'tight')



  '''
  idx = pd.date_range("2000-01-01",yesterday)
  WPD = WPD.reindex(idx,fill_value=0)

  
  # Probability of WPD to be higher than 10h
  WPD_hours = 10    
  WPD_prob = WPD.groupby(by=[WPD.index.month,WPD.index.day]).apply(lambda x: (len(x[x>=WPD_hours])/len(x))*100)

  WPD_prob.index = pd.to_datetime('2000' + '-' + WPD_prob.index.get_level_values(0).astype(str) + '-' +
                                   WPD_prob.index.get_level_values(1).astype(str),
                                   format='%Y-%m-%d')

  yearlabel_kws = dict(color='w',ha='center')
  pl2 = calplot.calplot(WPD_prob,cmap = 'Reds', figsize = (16, 3), 
                        textformat  ='{:.0f}',  yearlabel_kws = yearlabel_kws, suptitle_kws = suptitle_kws,
                        edgecolor='black', linewidth=2,
                        suptitle = row['loc'] + ", weather " + row['weather'] + "\nProbability of WPD to be equal or higher than 10 h")

  plt.savefig('../data/02_intermediate/WPD_prob_' + row['loc'] + '.png', bbox_inches = 'tight')
  print(time.time() - t)
  '''


# %%
## Temperature and precipitation line Graph

for index, row in df.iterrows():
 
  t = time.time()
  coords = {'lon':row['lon'], 'lat':row['lat']}

  queries = [] 

  for i in range(num_cores - 1):
    query_i = """ 
    DECLARE locations ARRAY<STRUCT<u_lat FLOAT64, u_lon FLOAT64>> DEFAULT [({},{})];
    DECLARE u_start_date DATE DEFAULT DATE('{}');
    DECLARE u_end_date DATE DEFAULT DATE('{}');
    DECLARE u_variables STRING DEFAULT 'min_temperature,max_temperature,total_precipitation';
    DECLARE uom STRING DEFAULT 'm';
    CALL `location360-datasets.historical_weather.historical_weather_daily_blend`(locations, u_start_date, u_end_date, u_variables, uom);
    """.format(coords['lat'], coords['lon'], seq_date[i].date() + timedelta(days=1), seq_date[i + 1].date()) 
    queries.append(query_i)
    
  results = p_map(runQuery, queries, **{"num_cpus": num_cores})

  df_t_pr = pd.concat(results)
  df_t_pr.index = pd.to_datetime(df_t_pr['date'],format='%Y-%m-%d')
  df_t_pr.sort_index(inplace=True)
  df_t_pr['avg_temp']=(df_t_pr.min_temperature+df_t_pr.max_temperature)/2

  start_day = pd.to_datetime('2017-01-01')
  end_day = pd.to_datetime(yesterday)

  df_t_pr_serie = df_t_pr.loc[start_day:end_day] 
  temp = df_t_pr_serie['avg_temp']
  prec = df_t_pr_serie['total_precipitation']
  prec[prec < 2] = 0

  suptitle_kws = dict(ha='center',size=22)
  pl_temp = calplot.calplot(temp,cmap = 'Reds', figsize = (16, 14), 
                        textformat  ='{:.0f}',  suptitle_kws = suptitle_kws,
                        edgecolor='black', linewidth=2,
                        suptitle = row['loc'] + ", weather " + row['weather'] +" \n Air temperature (°C)")

  plt.savefig('../data/02_intermediate/temp_hist_' + row['loc'] + '.png', bbox_inches = 'tight')  

  pl_prec = calplot.calplot(prec,cmap = 'Blues', figsize = (16, 14), 
                        textformat  ='{:.0f}',  suptitle_kws = suptitle_kws,
                        edgecolor='black', linewidth=2,
                        suptitle = row['loc'] + ", weather " + row['weather'] +" \n Total precipitation (mm)")

  plt.savefig('../data/02_intermediate/prec_hist_' + row['loc'] + '.png', bbox_inches = 'tight')  


# %%
prec = df_t_pr_serie['total_precipitation']
prec
# %%

  
# %%

   

  

# %%
start_day = pd.to_datetime('2017-01-01')
end_day = pd.to_datetime(yesterday)

df_t_pr_serie = df_t_pr.loc[start_day:end_day]

# %% [markdown]
# # Wetting-Period Duration
start_day = pd.to_datetime('2017-01-01')
end_day = pd.to_datetime(yesterday)
end_day
# %%

df_rh_night = df_rh['relative_humidity'].between_time('18:00', '06:00')
df_rh_night

# %%
df_rh_90 = df_rh_night[df_rh_night > 90]
df_rh_90

# %%
# Number of nights with RH above 90%  
WPD = df_rh_90.groupby(by=[df_rh_90.index.year, df_rh_90.index.month, df_rh_90.index.day]).count()
WPD

# %%
# %%
WPD.index = pd.to_datetime(WPD.index.get_level_values(0).astype(str) + '-' +
               WPD.index.get_level_values(1).astype(str) + '-' +
               WPD.index.get_level_values(2).astype(str),
               format='%Y-%m-%d')


# %%
start_day = pd.to_datetime('2017-01-01')
end_day = pd.to_datetime(yesterday)

# %%
WPD_serie = WPD.loc[start_day:end_day] 
WPD_serie

# %%
suptitle_kws = dict(ha='center',y=1.125,x=-0.025,size=22)
pl1 = calplot.calplot(WPD_serie,cmap = 'RdYlGn_r', figsize = (16, 12), 
                      textformat  ='{:.0f}',  suptitle_kws = suptitle_kws,
                      edgecolor='black', linewidth=2,
                      suptitle = "Number of hours at night with RH > 90% \n (Wetting-Period Duration)")

plt.savefig('teste_figure.png')

# %%
idx = pd.date_range("2000-01-01",yesterday)
WPD = WPD.reindex(idx,fill_value=0)
WPD
# %%
# Probability of duration WPD to be higher than 10h
WPD_hours = 10  
WPD_prob=WPD.groupby(by=[WPD.index.month,WPD.index.day]).apply(lambda x: len(x[x>WPD_hours])/len(x))


# %%
WPD_prob.index = pd.to_datetime('2000' + '-' + WPD_prob.index.get_level_values(0).astype(str) + '-' +
               WPD_prob.index.get_level_values(1).astype(str),
               format='%Y-%m-%d')



# %%
WPD_prob

# %%
yearlabel_kws = dict(color='w',ha='center')
pl1 = calplot.calplot(data = WPD_prob, cmap = 'Reds', figsize = (16, 8), suptitle = "Days with WPD >= 10h")

# %%
# Number of nights with at least 10 hours of RH above 90% (WPD>=10h) 
WPD_10h = df_rh_90.groupby(by=[df_rh_90.index.year, df_rh_90.index.month, df_rh_90.index.day]).count().loc[lambda x: x >= 10]
WPD_10h

# %%
WPD_10h.index = pd.to_datetime(WPD_10h.index.get_level_values(0).astype(str) + '-' +
               WPD_10h.index.get_level_values(1).astype(str) + '-' +
               WPD_10h.index.get_level_values(2).astype(str),
               format='%Y-%m-%d')
# %%


# %%
WPD_serie = WPD_10h.loc[start_day:end_day] 
WPD_serie

# %%

pl1 = calplot.calplot(data = WPD_serie, cmap = 'RdYlGn_r', figsize = (16, 8), suptitle = "Days with WPD >= 10h")


# %%
# Number of days with WPF >= 10
WPD_10h_10 = WPD_10h.groupby(by=[WPD_10h.index.year, WPD_10h.index.month]).count()
WPD_10h_10

# %%
WPD_10h_10.index=pd.to_datetime(WPD_10h_10.index.get_level_values(0).astype(str) + '-' +
               WPD_10h_10.index.get_level_values(1).astype(str),
               format='%Y-%m-%d')


# %%

weather_blend_hour="""
DECLARE locations ARRAY<STRUCT<u_lat FLOAT64, u_lon FLOAT64>> DEFAULT [({},{})];
DECLARE u_start_date DATE DEFAULT DATE('{}');
DECLARE u_end_date DATE DEFAULT DATE('{}');
DECLARE u_variables STRING DEFAULT 'relative_humidity';
DECLARE uom STRING DEFAULT 'm';
CALL `location360-datasets.historical_weather.historical_weather_hourly_blend`(locations, u_start_date, u_end_date, u_variables, uom);
""".format(coords['lat'], coords['long'],'2000-01-01',yesterday)

t = time.time()
df_wr = GBQQueryDataSet(weather_blend_hour, project='location360-datasets').load()
df_wr
print(time.time() - t)

## time: 1007.3402090072632

# %%

df_rh

# %%
df_wr.tail(30)
# %%
df_wr.describe()

# %%


# %%
df_wr.index = pd.to_datetime(df_wr['local_time'],format='%Y-%m-%d %H:%M:%S')
df_wr.index

# %%
df_wr



# %%
WPD_10h_10
# %%



# %%
WPD_10h.plot()


# %%

from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()



# %%
WPD_10h.plot()
# %%
WPD_10h.head()

# %%
df_rh_90.index.day

# %%
df_rh=df_wr.iloc['relative_humidity'].groupby(by=df_wr.index.day).mean()


# %% [markdown]
# # Precipitation

# %%
grid_ppt = """WITH ppt_cod AS (
  SELECT grid_id, geohash4, geom as ppt_geom FROM location360-datasets.historical_weather.twc_hires_precip_grids 
  ),
  point AS (
    SELECT ST_GEOGPOINT({}, {}) as pt_geom 
  )
SELECT grid_id, ppt_geom FROM ppt_cod, point WHERE geohash4 = ST_GEOHASH(point.pt_geom, 4) ORDER BY ST_DISTANCE(point.pt_geom, ppt_cod.ppt_geom)
LIMIT 5""".format(coords['long'], coords['lat'])

loc = GBQQueryDataSet(grid_ppt, project='location360-datasets').load()
loc

# %%
loc['ppt_geom'] = loc['ppt_geom'].apply(wkt.loads)
loc_gdf = gpd.GeoDataFrame(loc, geometry='ppt_geom')

# %% [markdown]
# ## Mapping 

map = folium.Map(location=[coords['lat'], coords['long']], zoom_start=12)
folium.Marker(
    location=[coords['lat'], coords['long']],
    popup="Location",
    icon=folium.Icon(color="green"),
).add_to(map)

loc_fol = folium.features.GeoJson(loc_gdf.to_json())
map.add_children(loc_fol)
map

# %% [markdown]
# ## Nearest grid
near_weather=loc_gdf.loc[[0]]
near_weather['grid_id'].squeeze()


# %%



# %%
# %%
loc_gdf.iloc[0]

# %%
float(coords['lat'])


#%% 
grid_id = 19432457
gridID_sql = """SELECT * FROM historical_weather.twc_cod_grids WHERE grid_id = '""" + str(grid_id) +"'"
gridID_sql

#%% 

#%% 


#%% 
#convert df['YEAR'] to int:
df['YEAR'] = df['YEAR'].astype(int)
df['YEAR'].head()


#%% [markdown]
# ## Feature engineering
df['Field_name']

# new Field_name with split value columns
new = df["Field_name"].str.split("#", n = 6, expand = True)
new

#%%
new_wrong = new.dropna()
new_wrong.drop(new_wrong.columns[2], axis=1, inplace=True)

new_wrong

#%%
new_right=new[new.loc[:,6].isnull()]
new_right.drop(new_right.columns[6], axis=1, inplace=True)

new_right

#%%
new = pd.concat([new_right,new_wrong])

#%%
new_right.type



#%%
#%% [markdown]
# ## Descriptive Statistical

#%%



# %%



#%% [markdown]
# ## Data Description 



#%%


#%%



#%%




#%%
  
#analyzing the dataset
advert_report = sv.analyze(df)
#display the report
advert_report.show_notebook(w=1500, h=300, scale=0.8)
  
#%%
sql_data
# %%
