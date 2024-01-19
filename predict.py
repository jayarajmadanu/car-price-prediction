from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logger
data = {
    'year': 2021,
    'km_driven': 15000,
    'fuel': 'Petrol',
    'seller_type': 'Individual',
    'transmission': 'Manual',
    'owner': 'First Owner'
}

predict = PredictPipeline()
res = predict.predict(data)

logger.info(f'prediction of data {data} is selling price {res}')