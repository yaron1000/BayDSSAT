import sys
if not '-m' in sys.argv:
    from .CSW.connect import CSWconnect
    from .Weather.get_wth import BayWeather
