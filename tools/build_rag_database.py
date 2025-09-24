import argparse

import lancedb
import torch
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def add_to_db(annotations: list[dict],
              model_name: str = 'Alibaba-NLP/gte-base-en-v1.5',
              text_name: str = 'llm_caption',
              db_path: str = '../data/rag.db', ) -> None:
    """
    Add annotations to the database
    :param annotations: list of annotations
    :param model_name: name of the model
    :param text_name: name of the text field
    :param db_path: path to the database
    :return: None
    """
    db = lancedb.connect(db_path)

    if text_name not in db.table_names(limit=10000):
        model = get_registry().get('sentence-transformers').create(name=model_name, device='cuda',
                                                                   model_kwargs={"torch_dtype": "bfloat16"},
                                                                   trust_remote_code=True)

        class TextEmbedding(LanceModel):
            text: str = model.SourceField()
            text_embedding: Vector(model.ndims()) = model.VectorField()
            id: int
            uid: str
            dataset: str
            video: str
            start_sec: float
            end_sec: float

        table = db.create_table(text_name, schema=TextEmbedding)
    else:
        table = db.open_table(text_name)

    for chunk in tqdm(chunks(annotations, 100_000), total=len(annotations) // 100_000):
        table.add(chunk, on_bad_vectors='fill')
    if len(table) > 1_000_000:
        table.create_index(metric='dot', vector_column_name='text_embedding')


def prepare_annotations(
        annotations: list[dict],
        text_name: str = 'llm_caption',
        dataset_name: str = 'coin') -> list[dict]:
    """
    Prepare annotations for the database
    :param annotations: list of annotations
    :param text_name: name of the text field
    :param dataset_name: name of the dataset
    :return:
    """
    return [{
        'text':      annotation[text_name] if annotation[text_name] is not None else '',
        'id':        annotation['id'],
        'uid':       dataset_name + '/' + str(annotation['id']),
        'dataset':   dataset_name,
        'video':     annotation['video'],
        'start_sec': annotation['start_sec'],
        'end_sec':   annotation['end_sec'],
    } for idx, annotation in enumerate(annotations)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default='../datasets/rag/openvid.db')
    parser.add_argument("--dataset", type=str, default='openvid')
    parser.add_argument("--annotations_path", type=str, default='../datasets/OpenVid-1M/data/openvid-1m.parquet')
    parser.add_argument('--caption_name', type=str, default='motion_caption')
    parser.add_argument("--model_name", type=str, default='Alibaba-NLP/gte-base-en-v1.5')
    args = parser.parse_args()

    annotations = torch.load(args.annotations_path)
    annotations = prepare_annotations(annotations, text_name=args.caption_name, dataset_name=args.dataset)
    add_to_db(annotations, model_name=args.model_name, text_name=args.caption_name, db_path=args.db_path)
