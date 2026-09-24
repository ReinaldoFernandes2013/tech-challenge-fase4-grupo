import pandas as pd
import random
from src.indexing.vector_store import OlistVectorStore
import warnings
warnings.filterwarnings('ignore')

def verify_mapping():
    df = pd.read_parquet('data/processed/olist_indexed_sample.parquet')
    sample_ids = random.sample(df['review_id'].dropna().tolist(), 5)
    
    print('=== Verificando Mapeamento review_id -> ChromaDB ===')
    print(f'Total de linhas no Parquet: {len(df)}')
    print(f'Total de review_ids UNICOS no Parquet: {df["review_id"].nunique()}')
    
    vstore = OlistVectorStore()
    store = vstore.get_store()
    
    success = True
    for rid in sample_ids:
        results = store.get(where={'review_id': rid})
        count = len(results['ids'])
        
        print(f'review_id: {rid} | Documentos encontrados no ChromaDB: {count}')
        if count != 1:
            success = False
            
    if success:
        print('\nSUCESSO: Cada review_id amostrado tem EXATAMENTE 1 documento no ChromaDB.')
    else:
        print('\nERRO: Algum review_id tem 0 ou multiplos documentos no ChromaDB.')

if __name__ == '__main__':
    verify_mapping()
