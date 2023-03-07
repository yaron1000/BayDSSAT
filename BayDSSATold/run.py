## Domino runs 
from domino import Domino

domino = Domino(
    "Field_Trialing_LATAM/FieldTrialing_Latam",
    api_key=os.environ["DOMINO_USER_API_KEY"],
    host=os.environ["DOMINO_API_HOST"],
)

# Blocking: this will start the run and wait for the run to finish before returning the status of the run
domino_run = domino.runs_start_blocking(
    ["/mnt/code/WaterBalance/DSSATcal.py", 5], title="Running DSSAT calibration for soybean"
)
print(domino_run)
