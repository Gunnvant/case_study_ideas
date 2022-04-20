from utils import Aggregate
from config import TO, FROM

agg = Aggregate(origin=FROM,dest=TO)
agg.aggregate_write()