import google_api
from google_images_search import GoogleImagesSearch

API_KEY = google_api.API_KEY
CX = google_api.SEARCH_CX

gis = GoogleImagesSearch(API_KEY, CX)
gis_params = {
    'q': '',
    'num': 1,
    'fileType': 'jpg|gif|png',
    'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived',
    'safe': 'medium',
    'imgType': 'photo'
}
# this will search and download:
gis.search(search_params=gis_params, path_to_dir='./')