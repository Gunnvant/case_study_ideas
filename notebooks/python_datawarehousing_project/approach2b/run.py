from config import TO,FROM
from utils import FetchData

fetch = FetchData(origin=FROM,dest=TO)
fetch.get_data_dict()
fetch.make_table()