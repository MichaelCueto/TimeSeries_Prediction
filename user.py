from create_db import database
from timeseries_model import time_series
from params import root,less_import_series
root_model = 'modelos/modelos_tph'
def user():
    db = database(root=root,lista=less_import_series,time='time',target='Tph',N_IN=4,delta_time=400)
    db1 = db.crate_database()
    model = time_series(data=db1,target='Tph',root_model=root_model)
    pred = model.model_ts(EM=4000, M=50,save_model=True)    
    return pred

if __name__ == '__main__':
    pred = user()
    print(pred)