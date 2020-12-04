import requests

from OpenRiverCam import log


r = requests.get("http://portal/api/sites")
print(r.text)
