from BayDSSAT import CSWconnect, BayWeather

print('WaterBalance imported')

project = 'location360-datasets'
query = """
SELECT business_region FROM environmental_data_cube.growth_stage_predictions_soybean
"""

#query_wth = """
#DECLARE locations ARRAY<STRUCT<u_lat FLOAT64, u_lon FLOAT64>> DEFAULT [({},{})];
#    DECLARE u_start_date DATE DEFAULT DATE('{}');
#    DECLARE u_end_date DATE DEFAULT DATE('{}');
#    DECLARE u_variables STRING DEFAULT 'min_temperature,max_temperature,total_precipitation';
#    DECLARE uom STRING DEFAULT 'm';
#    CALL `location360-datasets.historical_weather.historical_weather_daily_blend`(locations, u_start_date, u_end_date, u_variables, uom);
#    """.format(coords['lat'], coords['lon'], seq_date[i].date() + timedelta(days=1), seq_date[i + 1].date())


df = CSWconnect(project).load(query)
print(df.head())
